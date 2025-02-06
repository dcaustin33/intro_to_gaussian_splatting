import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm

from splat.gaussians import Gaussians
from splat.render_engine.schema import PreprocessedGaussian
from splat.render_engine.utils import (
    compute_fov_from_focal,
    compute_radius_from_covariance_2d,
    invert_covariance_2d,
)
from splat.utils import extract_gaussian_weight, ndc2Pix

if torch.cuda.is_available():
    from splat.c import preprocessing
    from splat.render_engine.autograd import autograd_render_tile_cuda


class GaussianScene2(nn.Module):
    def __init__(self, gaussians: Gaussians):
        super().__init__()
        self.gaussians = gaussians
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_params(self) -> List[torch.Tensor]:
        return [
            self.gaussians.quaternions,
            self.gaussians.scales,
            self.gaussians.colors,
            self.gaussians.opacity,
        ]

    @staticmethod
    def compute_2d_covariance(
        points_homogeneous: torch.Tensor,
        covariance_3d: torch.Tensor,
        extrinsic_matrix: torch.Tensor,
        tan_fovX: float,
        tan_fovY: float,
        focal_x: float,
        focal_y: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make sure the extrinsic matrix has the translation in the last row"""

        points_camera_space = points_homogeneous @ extrinsic_matrix
        x = points_camera_space[:, 0] / points_camera_space[:, 2]
        y = points_camera_space[:, 1] / points_camera_space[:, 2]

        x = torch.clamp(x, -1.3 * tan_fovX, 1.3 * tan_fovX) * points_camera_space[:, 2]
        y = torch.clamp(y, -1.3 * tan_fovY, 1.3 * tan_fovY) * points_camera_space[:, 2]
        z = points_camera_space[:, 2]
        j = torch.zeros((points_camera_space.shape[0], 3, 3)).to(
            points_camera_space.device
        )
        j[:, 0, 0] = focal_x / z
        j[:, 0, 2] = -(focal_x * x) / (z**2)
        j[:, 1, 1] = focal_y / z
        j[:, 1, 2] = -(focal_y * y) / (z**2)

        # we assume our extrinsic matrix has the translation in the last row
        # so it is already transposed so we transpose back
        # overall formula for a normal extrinsic matrix is
        w = extrinsic_matrix[:3, :3]
        t = w @ j.transpose(1, 2)

        covariance2d = (
            t.transpose(1, 2)
            @ covariance_3d.transpose(1, 2)  # doesnt this not do anything?
            @ t
        )
        # scale by 0.3 for the covariance and numerical stability on the diagonal
        # this is a hack to make the covariance matrix more stable
        final_covariance_2d = torch.zeros_like(covariance2d)
        final_covariance_2d[:, 0, 0] = covariance2d[:, 0, 0] + 0.3
        final_covariance_2d[:, 1, 1] = covariance2d[:, 1, 1] + 0.3
        final_covariance_2d[:, 0, 1] = covariance2d[:, 0, 1]
        final_covariance_2d[:, 1, 0] = covariance2d[:, 1, 0]
        return final_covariance_2d[:, :2, :2], points_camera_space

    def filter_in_view(
        self, points_ndc: torch.Tensor, znear: float = 0.2
    ) -> torch.Tensor:
        """Filters those points that are too close to the camera"""
        truth_array = points_ndc[:, 2] > znear
        truth_array = truth_array & (points_ndc[:, 0] < 1.3)
        truth_array = truth_array & (points_ndc[:, 0] > -1.3)
        truth_array = truth_array & (points_ndc[:, 1] < 1.3)
        truth_array = truth_array & (points_ndc[:, 1] > -1.3)
        return truth_array

    def compute_tiles_touched(
        self,
        points_pixel_space: torch.Tensor,
        radii: torch.Tensor,
        tile_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """This computes how many tiles each point touches

        The calculation is figuring out how many tiles the x spans
        Then how many the y spans then multiplying them together
        """
        # x corresponds to height in images
        # y corresponds to width in images
        max_tile_x = math.ceil(width / tile_size) - 1
        max_tile_y = math.ceil(height / tile_size) - 1

        top_left_x = torch.clamp(
            ((points_pixel_space[:, 0] - radii) / tile_size).int(),
            max=max_tile_x,
            min=0,
        )
        top_left_y = torch.clamp(
            ((points_pixel_space[:, 1] - radii) / tile_size).int(),
            max=max_tile_y,
            min=0,
        )
        bottom_right_x = torch.clamp(
            ((points_pixel_space[:, 0] + radii) / tile_size).int(),
            max=max_tile_x,
            min=0,
        )
        bottom_right_y = torch.clamp(
            ((points_pixel_space[:, 1] + radii) / tile_size).int(),
            max=max_tile_y,
            min=0,
        )

        # now we get the spans we should not worry about
        truth_array = (
            (top_left_x > max_tile_x)
            | (top_left_y > max_tile_y)
            | (bottom_right_x < 0)
            | (bottom_right_y < 0)
        )
        span_x = torch.clamp((bottom_right_x + 1) - top_left_x, min=1)
        span_y = torch.clamp((bottom_right_y + 1) - top_left_y, min=1)
        span_x[truth_array] = 0
        span_y[truth_array] = 0

        return (
            span_x * span_y,
            [
                top_left_x.clamp(max=max_tile_x, min=0),
                top_left_y.clamp(max=max_tile_y, min=0),
            ],
            [
                bottom_right_x.clamp(max=max_tile_x, min=0),
                bottom_right_y.clamp(max=max_tile_y, min=0),
            ],
        )

    def preprocess(
        self,
        extrinsic_matrix: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
        focal_x: float,
        focal_y: float,
        width: float,
        height: float,
        tile_size: int = 16,
        use_cuda: bool = False,
    ) -> None:
        """
        Code to preprocess the Gaussians.
        We end with the means in pixel space, the 2D covariance, and the radius'

        Intrinsic matrix should we the opengl with the z sign on the 3rd column (zero indexed)
        (ie transposed from the orginal scratchapixel)

        Extrinsic matrix should already be transposed
        """
        fovX = compute_fov_from_focal(focal_x, width)
        fovY = compute_fov_from_focal(focal_y, height)

        tan_fovX = math.tan(fovX / 2)
        tan_fovY = math.tan(fovY / 2)

        points_homogeneous = self.gaussians.homogeneous_points
        if use_cuda:
            covariance3d = preprocessing.get_3d_covariance_matrix_cuda(
                self.gaussians.quaternions, self.gaussians.scales
            ).view(-1, 3, 3)
            covariance2d, points_camera_space = (
                preprocessing.get_2d_covariance_matrix_cuda(
                    points_homogeneous.contiguous(),
                    covariance3d,
                    extrinsic_matrix.to(self.device).contiguous(),
                    tan_fovX,
                    tan_fovY,
                    focal_x,
                    focal_y,
                )
            )
        else:
            covariance3d = self.gaussians.get_3d_covariance_matrix()
            covariance2d, points_camera_space = self.compute_2d_covariance(
                points_homogeneous,
                covariance3d,
                extrinsic_matrix.to(self.device),
                tan_fovX,
                tan_fovY,
                focal_x,
                focal_y,
            )

        covariance2d = covariance2d.view(-1, 2, 2)
        # Nx4 - using the openGL convention
        points_ndc = points_camera_space @ intrinsic_matrix.to(self.device)
        points_ndc.retain_grad()
        non_in_place_points_ndc = torch.zeros_like(points_ndc)
        non_in_place_points_ndc[:, :2] = points_ndc[:, :2] / points_ndc[:, 3].unsqueeze(
            1
        )
        non_in_place_points_ndc[:, 2] = points_ndc[:, 2]
        points_in_view_bool_array = self.filter_in_view(non_in_place_points_ndc)
        final_points_ndc = non_in_place_points_ndc[points_in_view_bool_array]
        covariance2d = covariance2d[points_in_view_bool_array]
        color = self.gaussians.colors[points_in_view_bool_array].to(self.device)  # nx3
        opacity = self.gaussians.opacity[points_in_view_bool_array].to(self.device)

        inverted_covariance_2d = invert_covariance_2d(covariance2d)
        radius = compute_radius_from_covariance_2d(covariance2d)

        points_pixel_coords_x = ndc2Pix(final_points_ndc[:, 0], dimension=width).view(
            -1, 1
        )
        points_pixel_coords_y = ndc2Pix(final_points_ndc[:, 1], dimension=height).view(
            -1, 1
        )
        final_mean = torch.cat(
            [
                points_pixel_coords_x,
                points_pixel_coords_y,
                final_points_ndc[:, 2].view(-1, 1),
            ],
            dim=1,
        )
        tiles_touched, top_left, bottom_right = self.compute_tiles_touched(
            final_mean[:, 0:2], radius, tile_size, height, width
        )

        top_left = torch.stack([top_left[0], top_left[1]]).to(self.device)
        bottom_right = torch.stack([bottom_right[0], bottom_right[1]]).to(self.device)
        return PreprocessedGaussian(
            means_3d=final_mean,
            covariance_2d=covariance2d,
            radius=radius,
            inverted_covariance_2d=inverted_covariance_2d,
            tiles_touched=tiles_touched,
            top_left=top_left,
            bottom_right=bottom_right,
            color=color,
            opacity=opacity,
        )

    def create_key_to_tile_map(
        self,
        array: torch.Tensor,
        preprocessed_gaussians: PreprocessedGaussian,
    ) -> torch.Tensor:
        """
        Create a map from the tile to the gaussians that touch it.

        Array is a nx2 tensor where n is the cumulative sum of the tiles touched.
        Every entry for each gaussian should correspond to a tile touched.
        In this function we are denoting the tiles touched by the gaussian
        by the top left and bottom right of the tile.

        dims are:
        x_index, y_index, z_depth, gaussian_index
        """
        start_idx = 0
        for idx in range(len(preprocessed_gaussians.tiles_touched)):
            num_tiles = preprocessed_gaussians.tiles_touched[idx]
            if num_tiles == 0:
                continue
            old_starting_idx = start_idx

            # Get the tile coordinates for this gaussian
            top_left = [
                preprocessed_gaussians.top_left[0][idx],
                preprocessed_gaussians.top_left[1][idx],
            ]
            bottom_right = [
                preprocessed_gaussians.bottom_right[0][idx],
                preprocessed_gaussians.bottom_right[1][idx],
            ]
            z_depth = preprocessed_gaussians.means_3d[idx, 2]

            # Fill in array entries for each tile this gaussian touches
            for x in range(int(top_left[0]), int(bottom_right[0]) + 1):
                for y in range(int(top_left[1]), int(bottom_right[1]) + 1):
                    array[start_idx] = torch.tensor(
                        [x, y, z_depth, idx], device=array.device
                    )
                    start_idx += 1
            assert start_idx == old_starting_idx + num_tiles
        return array

    def get_start_idx(
        self, array: torch.Tensor, total_x_tiles: int, total_y_tiles: int
    ) -> int:
        """
        Function to get where the start of the idx for the tile is.

        Apparently we can use torch.where and torch.unique_consecutive for this
        """
        # create a total_tiles_x x total_tiles_y array where the input is the part in the array where it starts
        # then we can just take the argmax of that array to get the start idx
        array_map = torch.ones((total_y_tiles, total_x_tiles), device=array.device) * -1

        for idx in tqdm.tqdm(range(len(array))):
            tile_x = array[idx, 0].int().item()
            tile_y = array[idx, 1].int().item()
            array_map[tile_y, tile_x] = (
                idx if array_map[tile_y, tile_x] == -1 else array_map[tile_y, tile_x]
            )
        return array_map

    def render_pixel(
        self,
        x_value: int,
        y_value: int,
        mean_2d: torch.Tensor,
        covariance_2d: torch.Tensor,
        opacity: torch.Tensor,
        color: torch.Tensor,
        current_T: float,
        min_weight: float = 0.00001,
        verbose: bool = False,
    ) -> Optional[torch.Tensor]:
        """Uses alpha blending to render a pixel"""
        gaussian_strength = extract_gaussian_weight(
            mean_2d, torch.Tensor([x_value, y_value]), covariance_2d
        )
        alpha = gaussian_strength * torch.sigmoid(opacity)
        test_t = current_T * (1 - alpha)
        if verbose:
            print(
                f"x_value: {x_value}, y_value: {y_value}, gaussian_strength: {gaussian_strength}, alpha: {alpha}, test_t: {test_t}, mean_2d: {mean_2d}"
            )
        if test_t < min_weight:
            return
        return color * current_T * alpha, test_t

    def create_test_preprocessed_gaussians(self) -> PreprocessedGaussian:
        means_3d = torch.tensor([[0, 0, 1], [0, 30, 1]], device=self.device).float()
        colors = torch.tensor([[1, 0, 0], [0, 0, 1]], device=self.device).float()
        opacities = torch.tensor([1.0, 1.0], device=self.device)
        inverted_covariance_2d = torch.tensor(
            [[4, 2, 2, 4], [4, 2, 2, 4]], device=self.device, dtype=torch.float32
        ).view(2, 2, 2)
        covariance_2d = torch.inverse(inverted_covariance_2d)
        tiles_touched = torch.tensor([1, 1], device=self.device).int()

        # should be a 2xn array where first row is x coordinates and second row is y coordinates
        top_left = torch.tensor([[0, 0], [0, 1]], device=self.device)
        bottom_right = torch.tensor([[0, 0], [0, 1]], device=self.device)
        radius = torch.tensor([1, 1], device=self.device)
        return PreprocessedGaussian(
            means_3d=means_3d,
            covariance_2d=covariance_2d,
            color=colors,
            opacity=opacities,
            inverted_covariance_2d=inverted_covariance_2d,
            tiles_touched=tiles_touched,
            top_left=top_left,
            bottom_right=bottom_right,
            radius=radius,
        )

    def render(
        self,
        preprocessed_gaussians: PreprocessedGaussian,
        height: int,
        width: int,
        tile_size: int = 16,
        test: bool = False,
    ) -> None:
        """
        Rendering function - it will do all the steps to render
        the scene similar to the kernels the original authors use
        """
        if test:
            preprocessed_gaussians = self.create_test_preprocessed_gaussians()
            height = 32
            width = 16

        prefix_sum = torch.cumsum(
            preprocessed_gaussians.tiles_touched.to(torch.int64), dim=0
        )

        array = torch.zeros(
            (prefix_sum[-1], 4), device=self.device, dtype=torch.float32
        )
        # the first 32 bits will be the x_index of the tile
        # the next 32 bits will be the y_index of the tile
        # the last 32 bits will be the z depth of the gaussian

        array = self.create_key_to_tile_map(array, preprocessed_gaussians)
        # sort the array by the x and y coordinates
        sorted_indices = torch.argsort(
            array[:, 0] + array[:, 1] * 1e-4 + array[:, 2] * 1e-8
        )
        array = array[sorted_indices]
        starting_indices = self.get_start_idx(
            array, math.ceil(width / tile_size), math.ceil(height / tile_size)
        )

        image = torch.zeros((height, width, 3), device=self.device, dtype=torch.float32)
        t_values = torch.ones((height, width), device=self.device)
        done = torch.zeros((height, width), device=self.device, dtype=torch.bool)
        target_pixel_x = 2500
        target_pixel_y = 500
        target_tile_x = target_pixel_x // 16
        target_tile_y = target_pixel_y // 16

        for idx in tqdm.tqdm(range(len(array))):
            # you render for all the tiles the gaussian will touch
            gaussian_idx = array[idx, 3].int().item()
            tile_x = array[idx, 0]
            tile_y = array[idx, 1]

            starting_image_x = (tile_x * tile_size).int().item()
            starting_image_y = (tile_y * tile_size).int().item()

            if tile_x != target_tile_x or tile_y != target_tile_y:
                continue

            for x in range(starting_image_x, starting_image_x + tile_size):
                for y in range(starting_image_y, starting_image_y + tile_size):
                    # we should have a range here
                    if x >= width or y >= height:
                        continue
                    if x < 0 or y < 0:
                        continue
                    if done[y, x]:
                        continue
                    output = self.render_pixel(
                        x_value=x,
                        y_value=y,
                        mean_2d=preprocessed_gaussians.means_3d[gaussian_idx, :2],
                        covariance_2d=preprocessed_gaussians.covariance_2d[
                            gaussian_idx
                        ],
                        opacity=preprocessed_gaussians.opacity[gaussian_idx],
                        color=preprocessed_gaussians.color[gaussian_idx],
                        current_T=t_values[y, x],
                        verbose=(x == target_pixel_x and y == target_pixel_y),
                    )
                    if output is None:
                        done[y, x] = True
                        continue
                    image[y, x] += output[0]
                    t_values[y, x] = output[1]
        return image

    def render_cuda(
        self,
        preprocessed_gaussians: PreprocessedGaussian,
        height: int,
        width: int,
        tile_size: int = 16,
        test: bool = False,
    ) -> torch.Tensor:
        """
        Rendering function - it will do all the steps to render
        the scene similar to the kernels the original authors use
        """
        if test:
            preprocessed_gaussians = self.create_test_preprocessed_gaussians()
            height = 32
            width = 16

        prefix_sum = torch.cumsum(
            preprocessed_gaussians.tiles_touched.to(torch.int64), dim=0
        )

        array = torch.zeros(
            (prefix_sum[-1], 4), device=self.device, dtype=torch.float32
        )
        # the first 32 bits will be the x_index of the tile
        # the next 32 bits will be the y_index of the tile
        # the last 32 bits will be the z depth of the gaussian
        array = preprocessing.create_key_to_tile_map_cuda(
            array,
            preprocessed_gaussians.means_3d.contiguous(),
            preprocessed_gaussians.top_left.contiguous(),
            preprocessed_gaussians.bottom_right.contiguous(),
            prefix_sum,
        )
        # sort the array by the x and y coordinates
        # First, sort by 'z' coordinate
        _, indices = torch.sort(array[:, 2], stable=True)
        array = array[indices]

        # Then, sort by 'y' coordinate
        _, indices = torch.sort(array[:, 1], stable=True)
        array = array[indices]

        # Finally, sort by 'x' coordinate
        _, indices = torch.sort(array[:, 0], stable=True)
        array = array[indices]
        starting_indices = preprocessing.get_start_idx_cuda(
            array.contiguous(),
            math.ceil(width / tile_size),
            math.ceil(height / tile_size),
        )

        image = torch.zeros((height, width, 3), device=self.device, dtype=torch.float32)

        tile_indices = array[:, 0:2].int()
        array_indices = array[:, 3].int()
        starting_indices = starting_indices.int()
        final_tile_indices = (
            tile_indices[:, 1] * starting_indices.shape[1] + tile_indices[:, 0]
        )

        image = torch.zeros((height, width, 3), device=self.device, dtype=torch.float32)
        tile_size = 16

        image = autograd_render_tile_cuda.apply(
            tile_size,
            preprocessed_gaussians.means_3d.contiguous(),
            preprocessed_gaussians.color.contiguous(),
            preprocessed_gaussians.opacity.contiguous(),
            preprocessed_gaussians.inverted_covariance_2d.contiguous(),
            image.contiguous(),
            starting_indices.contiguous(),
            final_tile_indices.contiguous(),
            array_indices.contiguous(),
            height,
            width,
            len(preprocessed_gaussians.tiles_touched),
            array.shape[0],
        )
        image = image.clamp(0, 1)
        return image

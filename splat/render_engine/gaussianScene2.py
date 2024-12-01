import math
import time
from typing import Optional, Tuple

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
    from splat.c import render_tile_cuda


class GaussianScene2(nn.Module):
    def __init__(self, gaussians: Gaussians):
        super().__init__()
        self.gaussians = gaussians
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_2d_covariance(
        self,
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

        j = torch.zeros((points_camera_space.shape[0], 2, 3)).to(self.device)
        j[:, 0, 0] = focal_x / points_camera_space[:, 2]
        j[:, 0, 2] = (focal_x * points_camera_space[:, 0]) / (
            points_camera_space[:, 2] ** 2
        )
        j[:, 1, 1] = focal_y / points_camera_space[:, 2]
        j[:, 1, 2] = (focal_y * points_camera_space[:, 1]) / (
            points_camera_space[:, 2] ** 2
        )

        # we assume our extrinsic matrix has the translation in the last row
        # so it is already transposed so we transpose back
        # overall formula for a normal extrinsic matrix is
        # J @ W @ covariance_3d @ W.T @ J.T
        covariance2d = (
            j
            @ extrinsic_matrix[:3, :3].T
            @ covariance_3d
            @ extrinsic_matrix[:3, :3]
            @ j.transpose(1, 2)
        )
        return covariance2d, points_camera_space

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
        max_tile_x = math.ceil(width / tile_size) - 1
        max_tile_y = math.ceil(height / tile_size) - 1

        top_left_x = torch.clamp(((points_pixel_space[:, 0] - radii) / tile_size).int(), max=max_tile_x, min=0)
        top_left_y = torch.clamp(((points_pixel_space[:, 1] - radii) / tile_size).int(), max=max_tile_y, min=0)
        bottom_right_x = torch.clamp(((points_pixel_space[:, 0] + radii) / tile_size).int(), max=max_tile_x, min=0)
        bottom_right_y = torch.clamp(((points_pixel_space[:, 1] + radii) / tile_size).int(), max=max_tile_y, min=0)

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

        points_homogeneous = torch.cat(
            [
                self.gaussians.points.to(self.device),
                torch.ones(self.gaussians.points.shape[0], 1).to(self.device),
            ],
            dim=1,
        )  # Nx4

        covariance3d = self.gaussians.get_3d_covariance_matrix().to(self.device)
        covariance2d, points_camera_space = self.compute_2d_covariance(
            points_homogeneous,
            covariance3d,
            extrinsic_matrix.to(self.device),
            tan_fovX,
            tan_fovY,
            focal_x,
            focal_y,
        )
        # Nx4 - using the openGL convention
        points_ndc = points_camera_space @ intrinsic_matrix.to(self.device)
        points_ndc = points_ndc[:, :3] / points_ndc[:, 3].unsqueeze(1)  # nx3
        points_in_view_bool_array = self.filter_in_view(points_ndc)
        points_ndc = points_ndc[points_in_view_bool_array]
        covariance2d = covariance2d[points_in_view_bool_array]
        color = self.gaussians.colors[points_in_view_bool_array].to(self.device)  # nx3
        opacity = self.gaussians.opacity[points_in_view_bool_array].to(self.device)

        inverted_covariance_2d = invert_covariance_2d(covariance2d)
        radius = compute_radius_from_covariance_2d(covariance2d)

        points_pixel_coords_x = ndc2Pix(points_ndc[:, 0], dimension=width)
        points_pixel_coords_y = ndc2Pix(points_ndc[:, 1], dimension=height)
        points_ndc[:, 0] = points_pixel_coords_x
        points_ndc[:, 1] = points_pixel_coords_y
        tiles_touched, top_left, bottom_right = self.compute_tiles_touched(
            points_ndc[:, 0:2], radius, tile_size, height, width
        )

        return PreprocessedGaussian(
            means_3d=points_ndc,
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
        array_map = torch.ones((total_x_tiles, total_y_tiles), device=array.device) * -1
        for idx in range(len(array)):
            tile_x = array[idx, 0].int().item()
            tile_y = array[idx, 1].int().item()
            array_map[tile_x, tile_y] = (
                idx if array_map[tile_x, tile_y] == -1 else array_map[tile_x, tile_y]
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
    ) -> Optional[torch.Tensor]:
        """Uses alpha blending to render a pixel"""
        gaussian_strength = extract_gaussian_weight(
            mean_2d, torch.Tensor([x_value, y_value]), covariance_2d
        )
        alpha = gaussian_strength * torch.sigmoid(opacity)
        test_t = current_T * (1 - alpha)
        if test_t < min_weight:
            return
        return color * current_T * alpha, test_t

    def render(
        self,
        preprocessed_gaussians: PreprocessedGaussian,
        height: int,
        width: int,
        tile_size: int = 16,
    ) -> None:
        """
        Rendering function - it will do all the steps to render
        the scene similar to the kernels the original authors use
        """
        print("starting sum")
        prefix_sum = torch.cumsum(preprocessed_gaussians.tiles_touched, dim=0)
        print("ending sum")

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

        image = torch.zeros((width, height, 3), device=self.device, dtype=torch.float32)
        t_values = torch.ones((width, height), device=self.device)
        done = torch.zeros((width, height), device=self.device, dtype=torch.bool)

        for idx in tqdm.tqdm(range(len(array))):
            # you render for all the tiles the gaussian will touch
            gaussian_idx = array[idx, 3].int().item()
            tile_x = array[idx, 0]
            tile_y = array[idx, 1]

            starting_image_x = (tile_x * tile_size).int().item()
            starting_image_y = (tile_y * tile_size).int().item()

            for x in range(starting_image_x, starting_image_x + tile_size):
                for y in range(starting_image_y, starting_image_y + tile_size):
                    # we should have a range here
                    if x >= width or y >= height:
                        continue
                    if x < 0 or y < 0:
                        continue
                    if done[x, y]:
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
                        current_T=t_values[x, y],
                    )
                    if output is None:
                        done[x, y] = True
                        continue
                    image[x, y] += output[0]
                    t_values[x, y] = output[1]
        return image

    def render_cuda(
        self,
        preprocessed_gaussians: PreprocessedGaussian,
        height: int,
        width: int,
        tile_size: int = 16,
    ) -> None:
        """
        Rendering function - it will do all the steps to render
        the scene similar to the kernels the original authors use
        """
        print("starting sum")
        # preprocessed_gaussians.tiles_touched = preprocessed_gaussians.tiles_touched[:100]
        prefix_sum = torch.cumsum(preprocessed_gaussians.tiles_touched, dim=0)
        print("ending sum")

        print(prefix_sum[-1], preprocessed_gaussians.tiles_touched.shape[0])
        print(math.ceil(width / tile_size), math.ceil(height / tile_size))

        array = torch.zeros(
            (prefix_sum[-1], 4), device=self.device, dtype=torch.float16
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
        print(image.device)
        print("Starting render")

        tile_indices = array[:, 0:2].int()
        array_indices = array[:, 3].int()
        starting_indices = starting_indices.int()

        image = render_tile_cuda.render_tile_cuda(
            tile_size,
            preprocessed_gaussians.means_3d.contiguous(),
            preprocessed_gaussians.color.contiguous(),
            preprocessed_gaussians.opacity.contiguous(),
            preprocessed_gaussians.inverted_covariance_2d.contiguous(),
            image.contiguous(),
            starting_indices.contiguous(),
            tile_indices.contiguous(),
            array_indices.contiguous(),
            height,
            width,
            len(preprocessed_gaussians.tiles_touched),
        )
        return image
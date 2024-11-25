import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from splat.gaussians import Gaussians
from splat.render_engine.schema import PreprocessedGaussian
from splat.render_engine.utils import (
    compute_fov_from_focal,
    compute_radius_from_covariance_2d,
    invert_covariance_2d,
)
from splat.utils import ndc2Pix, extract_gaussian_weight


class GaussianScene2(nn.Module):
    def __init__(self, gaussians: Gaussians):
        super().__init__()
        self.gaussians = gaussians

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

        # I AM NOT SURE IF THIS IS RIGHT - I THINK THIS IS THE VIEW MATRIX
        points_camera_space = points_homogeneous @ extrinsic_matrix
        x = points_camera_space[:, 0] / points_camera_space[:, 2]
        y = points_camera_space[:, 1] / points_camera_space[:, 2]
        x = torch.clamp(x, -1.3 * tan_fovX, 1.3 * tan_fovX) * points_camera_space[:, 2]
        y = torch.clamp(y, -1.3 * tan_fovY, 1.3 * tan_fovY) * points_camera_space[:, 2]

        j = torch.zeros((points_camera_space.shape[0], 2, 3))
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
        return covariance2d

    def filter_in_view(
        self, points_ndc: torch.Tensor, znear: float = 0.2
    ) -> torch.Tensor:
        """Filters those points that are too close to the camera"""
        return points_ndc[:, 2] > znear

    def compute_tiles_touched(
        self, points_pixel_space: torch.Tensor, radii: torch.Tensor, tile_size: int
    ) -> torch.Tensor:
        """This computes how many tiles each point touches

        The calculation is figuring out how many tiles the x spans
        Then how many the y spans then multiplying them together
        """
        top_left_x = max((points_pixel_space - radii) / tile_size, 0)
        top_left_y = max((points_pixel_space - radii) / tile_size, 0)
        bottom_right_x = min((points_pixel_space + radii) / tile_size, tile_size - 1)
        bottom_right_y = min((points_pixel_space + radii) / tile_size, tile_size - 1)

        span_x = max(bottom_right_x - top_left_x, 1)
        span_y = max(bottom_right_y - top_left_y, 1)
        return (
            span_x * span_y,
            [top_left_x, top_left_y],
            [bottom_right_x, bottom_right_y],
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

        Intrinsic matrix should we the opengl with the -1 on the 3rd column
        (ie transposed from the orginal scratchapixel)
        """
        fovX = compute_fov_from_focal(focal_x, width)
        fovY = compute_fov_from_focal(focal_y, height)

        tan_fovX = math.tan(fovX / 2)
        tan_fovY = math.tan(fovY / 2)

        points_homogeneous = torch.cat(
            [self.gaussians.points, torch.ones(self.gaussians.points.shape[0], 1)],
            dim=1,
        )  # Nx4

        covariance3d = self.gaussians.covariance_3d
        covariance2d, points_camera_space = self.compute_2d_covariance(
            points_homogeneous,
            covariance3d,
            extrinsic_matrix,
            tan_fovX,
            tan_fovY,
            focal_x,
            focal_y,
        )
        # Nx4 - using the openGL convention
        points_ndc = points_camera_space @ intrinsic_matrix
        points_ndc = points_ndc[:, :3] / points_ndc[:, 3:4]  # Nx3
        points_in_view_bool_array = self.filter_in_view(points_ndc)
        points_ndc = points_ndc[points_in_view_bool_array]
        covariance2d = covariance2d[points_in_view_bool_array]
        color = self.gaussians.colors[points_in_view_bool_array] # nx3
        opacity = self.gaussians.opacity[points_in_view_bool_array]

        inverted_covariance_2d = invert_covariance_2d(covariance2d)
        radius = compute_radius_from_covariance_2d(covariance2d)

        tiles_touched, top_left, bottom_right = self.compute_tiles_touched(
            radius, tile_size
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
            opacity=opacity
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
                
            # Get the tile coordinates for this gaussian
            top_left = preprocessed_gaussians.top_left[idx]
            bottom_right = preprocessed_gaussians.bottom_right[idx]
            z_depth = preprocessed_gaussians.means_3d[idx, 2]
            
            # Fill in array entries for each tile this gaussian touches
            for x in range(int(top_left[0]), int(bottom_right[0]) + 1):
                for y in range(int(top_left[1]), int(bottom_right[1]) + 1):
                    array[start_idx] = torch.tensor([x, y, z_depth, idx], device=array.device)
                    start_idx += 1
                    
        return array
            

    def render_pixel(
        self,
        x_value: int,
        y_value:int,
        mean_2d: torch.Tensor,
        covariance_2d: torch.Tensor,
        opacity: torch.Tensor,
        color: torch.Tensor,
        current_T: float,
        min_weight: float = .01
    ) -> torch.Tensor:
        """Uses alpha blending to render a pixel"""
        gaussian_strength = extract_gaussian_weight(
            mean_2d, 
            torch.Tensor([x_value, y_value]),
            covariance_2d
        )
        alpha = gaussian_strength * torch.sigmoid(opacity)
        test_t = current_T * (1-alpha)
        if test_t < min_weight:
            return
        return color * current_T * alpha, test_t 

    def render(
        self, preprocessed_gaussians: PreprocessedGaussian, height: int, width: int, tile_size: int=16
    ) -> None:
        """
        Rendering function - it will do all the steps to render
        the scene similar to the kernels the original authors use
        """

        prefix_sum = torch.cumsum(preprocessed_gaussians.tiles_touched, dim=0)
        array = torch.zeros((prefix_sum[-1], 4), device=self.device, dtype=torch.float64)
        # the first 32 bits will be the x_index of the tile
        # the next 32 bits will be the y_index of the tile
        # the last 32 bits will be the z depth of the gaussian
        # the last 32 bits will be the gaussia idx

        array = self.create_key_to_tile_map(
            array, preprocessed_gaussians
        )

        # sort the array by the x and y coordinates
        sorted_indices = torch.argsort(array[:, 0] + array[:, 1] * 1e-4 + array[:, 2] * 1e-8)
        array = array[sorted_indices]
        
        covariance_2d = preprocessed_gaussians.covariance_2d[sorted_indices]
        
        image = torch.zeros((height, width, 3), device=self.device, dtype=torch.float32)
        t_values = torch.ones((height, width), device=self.device)
        done = torch.zeros((height, width), device=self.device, dtype=torch.float32)
        
        for idx in range(len(array)):
            # you render for all the tiles the gaussian will touch
            gaussian_idx = array[idx, 3]
            tile_x = array[idx, 0]
            tile_y = array[idx, 1]
            
            starting_image_x = tile_x * tile_size
            starting_image_y = tile_y * tile_size
            
            for x in range(starting_image_x, starting_image_x + tile_size):
                for y in range(starting_image_y, starting_image_y + tile_size):
                    # we should have a range here
                    if done[x, y]:
                        continue
                    output = self.render_pixel(
                        x_value=x,
                        y_value=y,
                        mean_2d=preprocessed_gaussians.means_3d[gaussian_idx, :2],
                        covariance_2d=covariance_2d,
                        opacity=preprocessed_gaussians.opacity[gaussian_idx],
                        color=preprocessed_gaussians.color[gaussian_idx],
                        current_T=t_values[x, y]
                    )
                    if output is None:
                        done[x, y] = True
                        continue
                    image[x, y] += output[0]
                    t_values[x, y] = output[1]
        
        

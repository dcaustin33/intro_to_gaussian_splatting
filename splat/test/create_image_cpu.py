""" 
This file does not have anything to do with ordering of the gaussians.
Renders on the CPU and uses pytorch backprop.
"""


import math
from dataclasses import dataclass

import torch
from matplotlib import pyplot as plt

from splat.render_engine.utils import compute_fov_from_focal
from splat.utils import build_rotation, get_extrinsic_matrix, getIntrinsicMatrix


@dataclass
class Gaussian:
    mean: torch.Tensor
    covariance: torch.Tensor
    color: torch.Tensor
    opacity: float
    
    @property
    def homogeneous_points(self):
        return torch.cat([self.mean, torch.ones(1, 1)], dim=1)
    
@dataclass
class Camera:
    focal_x: torch.Tensor
    focal_y: torch.Tensor
    c_x: torch.Tensor
    c_y: torch.Tensor
    width: torch.Tensor
    height: torch.Tensor
    camera_rotation: torch.Tensor
    camera_translation: torch.Tensor

    @property
    def intrinsic_matrix(self):
        return getIntrinsicMatrix(
            self.focal_x, 
            self.focal_y, 
            self.width,
            self.height
        )

    @property
    def extrinsic_matrix(self):
        return get_extrinsic_matrix(build_rotation(self.camera_rotation), self.camera_translation).T

    @property
    def fovX(self):
        return compute_fov_from_focal(self.focal_x, self.width)

    @property
    def fovY(self):
        return compute_fov_from_focal(self.focal_y, self.height)

    @property
    def tan_fovX(self):
        return math.tan(self.fovX / 2)

    @property
    def tan_fovY(self):
        return math.tan(self.fovY / 2)


from typing import Tuple


def ndc2Pix(points: torch.Tensor, dimension: int) -> torch.Tensor:
    """
    Convert points from NDC to pixel coordinates
    """
    return (points + 1) * (dimension - 1) * 0.5

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
    j = torch.zeros((points_camera_space.shape[0], 3, 3))
    j[:, 0, 0] = focal_x / z
    j[:, 0, 2] = -(focal_x * x) / (z ** 2)
    j[:, 1, 1] = focal_y / z
    j[:, 1, 2] = -(focal_y * y) / (z ** 2)

    # we assume our extrinsic matrix has the translation in the last row
    # so it is already transposed so we transpose back
    # overall formula for a normal extrinsic matrix is
    # J @ W @ covariance_3d @ W.T @ J.T
    w = extrinsic_matrix[:3, :3]
    t = w @ j.transpose(1, 2)
    covariance2d = (
        t.transpose(1, 2)
        @ covariance_3d.transpose(1, 2) # doesnt this not do anything?
        @ t
    )
    # scale by 0.3 for the covariance and numerical stability on the diagonal
    # this is a hack to make the covariance matrix more stable
    covariance2d[:, 0, 0] = covariance2d[:, 0, 0] + 0.3
    covariance2d[:, 1, 1] = covariance2d[:, 1, 1] + 0.3
    return covariance2d[:, :2, :2], points_camera_space


def extract_gaussian_weight(
    pixel: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor, pdb: bool = False
) -> torch.Tensor:
    """
    Use the covariance matrix to extract the weight of the point

    Args:
        mean: 1x2 tensor
        covariance: 2x2 tensor
    """
    diff = pixel - mean
    diff = diff.unsqueeze(0)
    gaussian_weight = -0.5 * torch.matmul(diff, torch.matmul(inv_covariance, diff.transpose(1, 2)))
    actual_weight = torch.exp(gaussian_weight)
    return actual_weight, gaussian_weight

def render_pixel(
    x_value: int,
    y_value: int,
    mean_2d: torch.Tensor,
    inv_covariance_2d: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    current_T: float,
    min_weight: float = 0.00001,
    verbose: bool = False,
):
    """Uses alpha blending to render a pixel"""
    gaussian_strength, exponent_weight = extract_gaussian_weight(
        torch.Tensor([x_value, y_value]), mean_2d, inv_covariance_2d
    )
    alpha = gaussian_strength * torch.sigmoid(opacity)
    test_t = current_T * (1 - alpha)
    if verbose:
        print(
            f"x_value: {x_value}, y_value: {y_value}, gaussian_strength: {gaussian_strength}, alpha: {alpha}, test_t: {test_t}, mean_2d: {mean_2d}"
        )
    if test_t < min_weight:
        return
    return color * current_T * alpha, test_t, current_T, gaussian_strength, exponent_weight


def create_image(camera: Camera, gaussian: Gaussian, height: int, width: int):
    points_camera_space = gaussian.homogeneous_points @ camera.extrinsic_matrix
    points_pixel_space = points_camera_space @ camera.intrinsic_matrix
    points_pixel_space_transformed = points_pixel_space[:, :2] / points_pixel_space[:, 3:4]
    pixel_x = ndc2Pix(points_pixel_space_transformed[:, 0], camera.width).view(-1, 1)
    pixel_y = ndc2Pix(points_pixel_space_transformed[:, 1], camera.height).view(-1, 1)
    point_ndc = torch.cat([pixel_x, pixel_y], dim=1)
    covariance_2d = compute_2d_covariance(
        gaussian.homogeneous_points, 
        gaussian.covariance, 
        camera.extrinsic_matrix, 
        camera.tan_fovX, 
        camera.tan_fovY, 
        camera.focal_x, 
        camera.focal_y
    )[0]
    inverted_covariance_2d = torch.linalg.inv(covariance_2d)
    image = torch.zeros((height, width, 3))
    print("point_ndc", point_ndc)
    print("gaussian.color", gaussian.color)
    print("gaussian.opacity", gaussian.opacity)
    for i in range(height):
        for j in range(width):
            image[i, j] = render_pixel(
                i, 
                j, 
                point_ndc[:, :2], 
                inverted_covariance_2d, 
                gaussian.opacity, 
                gaussian.color, 
                1.0
            )[0]
    return image, point_ndc
    
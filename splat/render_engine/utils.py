import math
from typing import Tuple

import torch

from splat.render_engine.schema import ClippingPlanes, Fov


def compute_intrinsic_matrix(
    clipping_planes: ClippingPlanes,
) -> torch.Tensor:
    """
    This is an intrinsic matrix for multiplying points

    This should be used points (nx4) @ intrinsic_matrix.T
    or intrinsic_matrix @ points (4xn)

    Args:
        near: near plane - all of these are calculated with fov
        far: far plane
        right: right plane
        left: left plane
        top: top plane
        bottom: bottom plane
    """
    return torch.Tensor(
        [
            [
                2
                * clipping_planes.near
                / (clipping_planes.right - clipping_planes.left),
                0,
                (clipping_planes.right + clipping_planes.left)
                / (clipping_planes.right - clipping_planes.left),
                0,
            ],
            [
                0,
                2
                * clipping_planes.near
                / (clipping_planes.top - clipping_planes.bottom),
                (clipping_planes.top + clipping_planes.bottom)
                / (clipping_planes.top - clipping_planes.bottom),
                0,
            ],
            [
                0,
                0,
                -(clipping_planes.far + clipping_planes.near)
                / (clipping_planes.far - clipping_planes.near),
                -2
                * clipping_planes.far
                * clipping_planes.near
                / (clipping_planes.far - clipping_planes.near),
            ],
            [0, 0, -1, 0],
        ]
    )


def compute_clipping_planes(
    fovX: float,
    fovY: float,
    near: float,
    far: float,
    width: float,
    height: float,
) -> ClippingPlanes:
    top = math.tan(fovY / 2) * near
    bottom = -top
    right = math.tan(fovX / 2) * (width / height) * near
    left = -right
    return ClippingPlanes(
        near=near, far=far, right=right, left=left, top=top, bottom=bottom
    )


def compute_fov_from_focal(focal: float, pixels: int) -> float:
    return 2 * math.atan(pixels / (2 * focal))


def compute_fovs(
    focal_x: float,
    focal_y: float,
    width: float,
    height: float,
) -> Fov:
    return Fov(
        fovX=compute_fov_from_focal(focal_x, width),
        fovY=compute_fov_from_focal(focal_y, height),
    )


def compute_radius(
    covariance_2d: torch.Tensor,
) -> torch.Tensor:
    return 3.0 * torch.sqrt(torch.det(covariance_2d))


def invert_covariance_2d(
    covariance_2d: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Covariance will ve a nX2X2 tensor"""
    cov = covariance_2d + epsilon
    determinant = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
    inverted_cov = torch.zeros_like(cov)
    multiplier = 1.0 / determinant
    inverted_cov[:, 0, 0] = cov[:, 1, 1] * multiplier
    inverted_cov[:, 0, 1] = -cov[:, 0, 1] * multiplier
    inverted_cov[:, 1, 0] = -cov[:, 1, 0] * multiplier
    inverted_cov[:, 1, 1] = cov[:, 0, 0] * multiplier
    return inverted_cov

def compute_radius_from_covariance_2d(
    covariance_2d: torch.Tensor,
    std_dev_multiplier: float = 3.0,
) -> torch.Tensor:
    """
    Computes the radius by using the trace of the 
    covariance matrix to find the eigenvalues
    """
    mid = 0.5 * (covariance_2d[:, 0, 0] + covariance_2d[:, 1, 1])
    det = covariance_2d[:, 0, 0] * covariance_2d[:, 1, 1] - covariance_2d[:, 0, 1] ** 2
    lambda1_intermediate = mid**2 - det
    lambda1 = mid + torch.sqrt(torch.clamp(mid**2 - det, min=0.1))
    lambda2 = mid - torch.sqrt(torch.clamp(mid**2 - det, min=0.1))
    max_lambda = torch.max(lambda1, lambda2)
    return std_dev_multiplier * torch.sqrt(max_lambda)

def ndc2Pix(points: torch.Tensor, height: float, width: float) -> torch.Tensor:
    """Convert points from NDC to pixel coordinates"""
    return (points + 1) * (height - 1) * 0.5
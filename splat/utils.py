import numpy as np
import torch

from typing import Tuple
import torch

def get_intrinsic_matrix(f_x: float, f_y: float, c_x: float, c_y: float) -> torch.Tensor:
    """
    Get the homogenous intrinsic matrix for the camera

    Args:
        f_x: focal length in x
        f_y: focal length in y
        c_x: principal point in x
        c_y: principal point in y
    """
    return torch.Tensor(
        [
            [f_x, 0, c_x, 0],
            [0, f_y, c_y, 0],
            [0, 0, 1, 0],
        ]
    )
    
def get_extrinsic_matrix(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Get the homogenous extrinsic matrix for the camera

    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    """
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def project_points(
    projection_matrix: torch.Tensor, points: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projection matrix is the intrinsic matrix a 3x4 matrix, points is a Nx3 matrix. 
    Returns a Nx2 matrix
    """
    points = torch.cat(
        [points, torch.ones(points.shape[0], 1, device=points.device)], dim=1
    )
    projected_points = torch.matmul(projection_matrix, points.t()).t()
    z_component = projected_points[:, 2].unsqueeze(1)
    projected_points = projected_points[:, :2] / projected_points[:, 2].unsqueeze(1)
    
    return projected_points, z_component


def getWorld2View(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    We should translate the points to camera coordinates, 
    however we will never use the 4th dimension it seems like so leaving for now
    """
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def extract_gaussian_weight(pixel: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    """
    Use the covariance matrix to extract the weight of the point

    Args:
        mean: 1x2 tensor
        covariance: 2x2 tensor
    """
    diff = pixel - mean
    inv_covariance = torch.inverse(covariance)
    return torch.exp(-0.5 * torch.matmul(diff, torch.matmul(inv_covariance, diff.t())))


if __name__ == "__main__":
    points = torch.Tensor(
        [
            [1.5, 3, 1],
        ]
    )

    covariance = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0.000000001],
            ]
        ]
    )
    extrinsic_matrix = torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    intrinsic_matrix = torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    pixel = torch.Tensor([1.5, 3])

    projected_points, projected_covariance = get_points_and_covariance(
        points, covariance, extrinsic_matrix, intrinsic_matrix
    )
    assert (projected_points[0,] == torch.Tensor([1.5, 3])).all()
    assert np.isclose(
        projected_covariance[0,].numpy(), np.array([[1, 0], [0, 1]])
    ).all()
    assert extract_weight(pixel, projected_points[0], projected_covariance[0]) == 1.0

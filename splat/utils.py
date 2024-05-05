import numpy as np
import torch

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
) -> torch.Tensor:
    """Projection matrix is a 3x4 matrix, points is a Nx3 matrix. Returns a Nx2 matrix"""
    points = torch.cat(
        [points, torch.ones(points.shape[0], 1, device=points.device)], dim=1
    )
    projected_points = torch.matmul(projection_matrix, points.t()).t()
    projected_points = projected_points[:, :2] / projected_points[:, 2].unsqueeze(1)
    return projected_points


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


def get_points_and_covariance(
    points: torch.Tensor,
    covariance_3d: torch.Tensor,
    extrinsic_matrix: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Given a set of points, we project to 2d and find their 2d covariance matrices

    Args:
        points: Nx3 tensor of points, will add a 4th dimension for homogeneous coordinates
        covariance_3d: Nx3x3 tensor of covariance matrices
        extrinsic_matrix: 4x4 tensor translates the points to camera coordinates but still in 3d
        intrinsic_matrix: 3x4 tensor that projects the points to 2d
    """
    points = torch.cat(
        [points, torch.ones(points.shape[0], 1, device=points.device)], dim=1
    )
    # results in a 4xN tensor
    points_in_camera_coords = torch.matmul(extrinsic_matrix, points.t()).T  # Nx4
    print(points.shape)
    # do not need to divide by 1 as this is always 1
    final_points_in_camera_coords = points_in_camera_coords[
        :, :3
    ] / points_in_camera_coords[:, 3].unsqueeze(1)
    # now we project to 2d
    projected_points = project_points(intrinsic_matrix, final_points_in_camera_coords)
    # now we find the covariance matrices in 2d
    projected_covariance = []

    f_x = intrinsic_matrix[0, 0]
    f_y = intrinsic_matrix[1, 1]

    # this makes it stored in column major order - something the original code does
    _W = getWorld2View(extrinsic_matrix[:3, :3], extrinsic_matrix[:3, 3]).transpose(
        0, 1
    )
    W = torch.Tensor(
        [
            [_W[0, 0], _W[1, 0], _W[2, 0]],
            [_W[0, 1], _W[1, 1], _W[2, 1]],
            [_W[0, 2], _W[1, 2], _W[2, 2]],
        ]
    )

    for i in range(covariance_3d.shape[0]):
        covariance = covariance_3d[i]
        camera_coords_x = points_in_camera_coords[i, 0]
        camera_coords_y = points_in_camera_coords[i, 1]
        camera_coords_z = points_in_camera_coords[i, 2]
        jacobian = torch.zeros((3, 3), device=points.device)
        jacobian[0, 0] = f_x / camera_coords_z
        jacobian[1, 1] = f_y / camera_coords_z
        jacobian[0, 2] = -f_x * camera_coords_x / (camera_coords_z**2)
        jacobian[1, 2] = -f_y * camera_coords_y / (camera_coords_z**2)
        T = torch.matmul(jacobian, W)
        final_variance = torch.matmul(T, torch.matmul(covariance, T.t()))
        projected_covariance.append(final_variance[:2, :2])
    return projected_points, torch.stack(projected_covariance)


def extract_weight(pixel: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
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

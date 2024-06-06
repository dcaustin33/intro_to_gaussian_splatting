import math
import os
from typing import Dict

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch.utils.cpp_extension import load_inline

from splat.read_colmap import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
)
from splat.schema import BasicPointCloud


def get_intrinsic_matrix(
    f_x: float, f_y: float, c_x: float, c_y: float
) -> torch.Tensor:
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
    points: torch.Tensor, intrinsic_matrix: torch.Tensor, extrinsic_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Project the points to the image plane

    Args:
        points: Nx3 tensor
        intrinsic_matrix: 3x4 tensor
        extrinsic_matrix: 4x4 tensor
    """
    homogeneous = torch.ones((4, points.shape[0]), device=points.device)
    homogeneous[:3, :] = points
    projected_to_camera_perspective = extrinsic_matrix @ homogeneous
    projected_to_image_plane = intrinsic_matrix @ projected_to_camera_perspective
    projected_points = projected_to_image_plane[:2, :] / projected_to_image_plane[
        2, :
    ].unsqueeze(1)
    x = projected_points[0, :]
    y = projected_points[1, :]
    return x, y


def extract_gaussian_weight(
    pixel: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor
) -> torch.Tensor:
    """
    Use the covariance matrix to extract the weight of the point

    Args:
        mean: 1x2 tensor
        covariance: 2x2 tensor
    """
    diff = pixel - mean
    inv_covariance = torch.inverse(covariance)
    return torch.exp(-0.5 * torch.matmul(diff, torch.matmul(inv_covariance, diff.t())))


def fetchPly(path: str):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def build_rotation(r: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def focal2fov(focal: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    return torch.Tensor([2 * math.atan(pixels / (2 * focal))])


def getWorld2View(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """This is the function to focus on as opposed to v2 below

    This take the rotation matrix and translation vector and returns the
    """
    Rt = torch.zeros((4, 4))

    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt.float()


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.t()  # Use .t() for transpose in PyTorch
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.inverse(Rt)  # Use torch.inverse for matrix inversion in PyTorch
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.inverse(C2W)
    return Rt.float()


def getProjectionMatrix(
    znear: torch.Tensor, zfar: torch.Tensor, fovX: torch.Tensor, fovY: torch.Tensor
) -> torch.Tensor:
    """
    znear: near plane set by user
    zfar: far plane set by user
    fovX: field of view in x, calculated from the focal length
    fovY: field of view in y, calculated from the focal length


    This is from the original repo.
    It uses the view to adjust the coordinates to the actual pixel dimensions
    It still retains the z componenet.
    This is the perspective projection matrix.
    When used in conjunction wih the world2view matrix, it will transform the points
    to the pixel coordinates.
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getIntinsicMatrix(
    focal_x: torch.Tensor,
    focal_y: torch.Tensor,
    height: torch.Tensor,
    width: torch.Tensor,
    znear: torch.Tensor = torch.Tensor([100.0]),
    zfar: torch.Tensor = torch.Tensor([0.001]),
) -> torch.Tensor:
    """
    Gets the internal perspective projection matrix

    znear: near plane set by user
    zfar: far plane set by user
    fovX: field of view in x, calculated from the focal length
    fovY: field of view in y, calculated from the focal length
    """
    fovX = torch.Tensor([2 * math.atan(width / (2 * focal_x))])
    fovY = torch.Tensor([2 * math.atan(height / (2 * focal_y))])

    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def read_camera_file(colmap_path: str) -> Dict:
    binary_path = os.path.join(colmap_path, "cameras.bin")
    text_path = os.path.join(colmap_path, "cameras.txt")
    if os.path.exists(binary_path):
        cameras = read_cameras_binary(binary_path)
    elif os.path.exists(text_path):
        cameras = read_cameras_text(text_path)
    else:
        raise ValueError
    return cameras


def read_image_file(colmap_path: str) -> Dict:
    binary_path = os.path.join(colmap_path, "images.bin")
    text_path = os.path.join(colmap_path, "images.txt")
    if os.path.exists(binary_path):
        images = read_images_binary(binary_path)
    elif os.path.exists(text_path):
        images = read_images_text(text_path)
    else:
        raise ValueError
    return images


def in_view_frustum(
    points: torch.Tensor, view_matrix: torch.Tensor, minimum_z: float = 0.2
) -> torch.Tensor:
    """
    Given a view matrix (transforming from world to camera coords) and a minimum
    z value, return a boolean tensor indicating whether the points aree in view.

    points is a Nx3 tensor and we return a N tensor indicating whether the point
    is in view.

    minimum_z is the minimum z set in the authors code
    """
    homogeneous = torch.ones((points.shape[0], 4), device=points.device)
    homogeneous[:, :3] = points
    projected_points = homogeneous @ view_matrix
    z_component = projected_points[:, 2]
    truth = z_component >= minimum_z
    return truth


def ndc2Pix(points: torch.Tensor, dimension: int) -> torch.Tensor:
    """
    Convert points from NDC to pixel coordinates
    """
    return (points + 1) * (dimension - 1) * 0.5


def compute_2d_covariance(
    points: torch.Tensor,
    extrinsic_matrix: torch.Tensor,
    covariance_3d: torch.Tensor,
    tan_fovY: torch.Tensor,
    tan_fovX: torch.Tensor,
    focal_x: torch.Tensor,
    focal_y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the 2D covariance matrix for each gaussian
    """
    points = torch.cat(
        [points, torch.ones(points.shape[0], 1, device=points.device)], dim=1
    )
    points_transformed = (points @ extrinsic_matrix)[:, :3]
    limx = 1.3 * tan_fovX
    limy = 1.3 * tan_fovY
    x = points_transformed[:, 0] / points_transformed[:, 2]
    y = points_transformed[:, 1] / points_transformed[:, 2]
    z = points_transformed[:, 2]
    x = torch.clamp(x, -limx, limx) * z
    y = torch.clamp(y, -limy, limy) * z

    J = torch.zeros((points_transformed.shape[0], 3, 3), device=covariance_3d.device)
    J[:, 0, 0] = focal_x / z
    J[:, 0, 2] = -(focal_x * x) / (z**2)
    J[:, 1, 1] = focal_y / z
    J[:, 1, 2] = -(focal_y * y) / (z**2)

    # transpose as originally set up for perspective projection
    # so we now transform back
    W = extrinsic_matrix[:3, :3].T

    return (J @ W @ covariance_3d @ W.T @ J.transpose(1, 2))[:, :2, :2]


def compute_gaussian_weight(
    pixel_coord: torch.Tensor,  # (1, 2) tensor
    point_mean: torch.Tensor,
    inverse_covariance: torch.Tensor,
) -> torch.Tensor:

    difference = point_mean - pixel_coord
    power = -0.5 * difference @ inverse_covariance @ difference.T
    return torch.exp(power).item()


def compute_inverted_covariance(covariance_2d: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse covariance matrix

    For a 2x2 matrix
    given as
    [[a, b],
     [c, d]]
     the determinant is ad - bc

    To get the inverse matrix reshuffle the terms like so
    and multiply by 1/determinant
    [[d, -b],
     [-c, a]] * (1 / determinant)
    """
    determinant = (
        covariance_2d[:, 0, 0] * covariance_2d[:, 1, 1]
        - covariance_2d[:, 0, 1] * covariance_2d[:, 1, 0]
    )
    determinant = torch.clamp(determinant, min=1e-3)
    inverse_covariance = torch.zeros_like(covariance_2d)
    inverse_covariance[:, 0, 0] = covariance_2d[:, 1, 1] / determinant
    inverse_covariance[:, 1, 1] = covariance_2d[:, 0, 0] / determinant
    inverse_covariance[:, 0, 1] = -covariance_2d[:, 0, 1] / determinant
    inverse_covariance[:, 1, 0] = -covariance_2d[:, 1, 0] / determinant
    return inverse_covariance


def compute_radius(covariance_2d: torch.Tensor) -> torch.Tensor:
    determinant = (
        covariance_2d[:, 0, 0] * covariance_2d[:, 1, 1]
        - covariance_2d[:, 0, 1] * covariance_2d[:, 1, 0]
    )
    midpoint = 0.5 * (covariance_2d[:, 0, 0] + covariance_2d[:, 1, 1])
    lambda1 = midpoint + torch.sqrt(midpoint**2 - determinant)
    lambda2 = midpoint - torch.sqrt(midpoint**2 - determinant)
    max_lambda = torch.max(lambda1, lambda2)
    radius = torch.ceil(2.5 * torch.sqrt(max_lambda))
    return radius


def compute_extent_and_radius(covariance_2d: torch.Tensor):
    mid = 0.5 * (covariance_2d[:, 0, 0] + covariance_2d[:, 1, 1])
    det = covariance_2d[:, 0, 0] * covariance_2d[:, 1, 1] - covariance_2d[:, 0, 1] ** 2
    intermediate_matrix = (mid * mid - det).view(-1, 1)
    intermediate_matrix = torch.cat(
        [intermediate_matrix, torch.ones_like(intermediate_matrix) * 0.1], dim=1
    )

    max_values = torch.max(intermediate_matrix, dim=1).values
    lambda1 = mid + torch.sqrt(max_values)
    lambda2 = mid - torch.sqrt(max_values)
    # now we have the eigenvalues, we can calculate the max radius
    max_radius = torch.ceil(3.0 * torch.sqrt(torch.max(lambda1, lambda2)))

    return max_radius


def load_cuda(cuda_src, cpp_src, funcs, opt=True, verbose=False):
    return load_inline(
        name="inline_ext",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=funcs,
        extra_cuda_cflags=["-O1"] if opt else [],
        verbose=verbose,
    )

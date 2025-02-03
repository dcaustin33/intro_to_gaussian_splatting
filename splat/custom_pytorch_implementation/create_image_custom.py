from typing import List

import torch

from splat.custom_pytorch_implementation.auto_functions import render_pixel_auto
from splat.custom_pytorch_implementation.covariance_derivatives import r_s_to_cov_2d
from splat.custom_pytorch_implementation.create_image_auto import (
    Camera,
    Gaussian,
    compute_2d_covariance,
    compute_j_w_matrix,
    ndc2Pix,
    render_pixel,
)
from splat.custom_pytorch_implementation.gaussian_weight_derivatives import (
    camera_space_to_pixel_space,
    mean_3d_to_camera_space,
    ndc_to_pixels,
    render_pixel_custom,
)
from splat.gaussians import Gaussians
from splat.render_engine.gaussianScene2 import GaussianScene2


def create_image_covariance_test_custom(
    camera: Camera, gaussian: Gaussian, height: int, width: int
):
    """
    Camera gives the internal and the external matrices.
    Gaussian_Covariance_Test gives the mean_2d, r, s, color, opacity
        Note the mean here has a third dimension for the depth needed for J
    height and width are the dimensions of the image
    """
    image = torch.zeros((height, width, 3))
    r = gaussian.r
    s = gaussian.s
    mean_2d = gaussian.mean_3d
    color = gaussian.color
    opacity = gaussian.opacity

    j_w_matrix = compute_j_w_matrix(camera, mean_2d)
    covariance_2d = r_s_to_cov_2d(r, s, j_w_matrix)
    for i in range(height):
        for j in range(width):
            # image[j, i] = render_pixel(
            #     i,
            #     j,
            #     mean_2d[:, :2],
            #     covariance_2d,
            #     opacity,
            #     color,
            #     1.0
            # )[0]
            image[j, i] = render_pixel_custom(
                torch.tensor([i, j]),
                mean_2d[:, :2],
                covariance_2d,
                opacity,
                color,
                torch.tensor(1.0),
            )[0]

    return image


def create_image_full_custom(
    camera: Camera,
    gaussian: Gaussian,
    height: int,
    width: int,
    image: torch.Tensor = None,
    current_Ts: torch.Tensor = None,
):
    if image is None:
        image = torch.zeros((height, width, 3))
    r = gaussian.r
    s = gaussian.s
    mean_3d = gaussian.homogeneous_points

    camera_space_mean = mean_3d_to_camera_space.apply(mean_3d, camera.extrinsic_matrix)
    pixel_space_mean = camera_space_to_pixel_space.apply(
        camera_space_mean, camera.intrinsic_matrix
    )
    new_pixel_space_mean = pixel_space_mean[:, :3] / pixel_space_mean[:, 3].unsqueeze(1)
    final_pixel_space_mean = torch.cat(
        [new_pixel_space_mean, pixel_space_mean[:, 3].unsqueeze(1)], dim=1
    )
    pixel_mean = ndc_to_pixels.apply(final_pixel_space_mean, [height, width])
    final_mean_2d = torch.cat([pixel_mean[:, :2], pixel_mean[:, 3].unsqueeze(1)], dim=1)

    color = gaussian.color
    opacity = gaussian.opacity
    j_w_matrix = compute_j_w_matrix(camera, gaussian.homogeneous_points)
    covariance_2d = r_s_to_cov_2d(r, s, j_w_matrix)
    for i in range(height):
        for j in range(width):
            image[j, i] = render_pixel_custom(
                torch.tensor([i, j]),
                final_mean_2d[:, :2],
                covariance_2d,
                opacity,
                color,
                current_Ts[i, j],
            )
    return image


def create_image_full_custom_multiple_gaussians(
    camera: Camera,
    gaussians: List[Gaussian],
    height: int,
    width: int,
    image: torch.Tensor = None,
):
    if image is None:
        image = torch.zeros((height, width, 3))
    all_final_means_2d = []
    all_r_s_to_cov_2d = []
    all_opacity = []
    all_color = []

    for gaussian in gaussians:
        r = gaussian.r
        s = gaussian.s
        mean_3d = gaussian.homogeneous_points

        camera_space_mean = mean_3d_to_camera_space.apply(
            mean_3d, camera.extrinsic_matrix
        )
        pixel_space_mean = camera_space_to_pixel_space.apply(
            camera_space_mean, camera.intrinsic_matrix
        )
        new_pixel_space_mean = pixel_space_mean[:, :3] / pixel_space_mean[
            :, 3
        ].unsqueeze(1)
        final_pixel_space_mean = torch.cat(
            [new_pixel_space_mean, pixel_space_mean[:, 3].unsqueeze(1)], dim=1
        )
        pixel_mean = ndc_to_pixels.apply(final_pixel_space_mean, [height, width])
        final_mean_2d = torch.cat(
            [pixel_mean[:, :2], pixel_mean[:, 3].unsqueeze(1)], dim=1
        )

        color = gaussian.color
        opacity = gaussian.opacity
        j_w_matrix = compute_j_w_matrix(camera, gaussian.homogeneous_points)
        covariance_2d = r_s_to_cov_2d(r, s, j_w_matrix)
        all_final_means_2d.append(final_mean_2d)
        all_r_s_to_cov_2d.append(covariance_2d)
        all_opacity.append(opacity)
        all_color.append(color)

    for i in range(height):
        for j in range(width):
            current_t = torch.tensor(1.0)
            for gaussian_index in range(len(gaussians)):
                color, current_t = render_pixel_custom(
                    torch.tensor([i, j]),
                    all_final_means_2d[gaussian_index][:, :2],
                    all_r_s_to_cov_2d[gaussian_index],
                    all_opacity[gaussian_index],
                    all_color[gaussian_index],
                    current_t,
                )
                image[j, i] += color[0]
    return image


def create_image_full_custom_multiple_gaussians_with_splat_gaussians(
    camera: Camera,
    gaussians: Gaussians,
    height: int,
    width: int,
    image: torch.Tensor = None,
    verbose: bool = False,
):
    device = gaussians.device
    if image is None:
        image = torch.zeros((height, width, 3)).to(device)
    all_final_means_2d = []
    all_r_s_to_cov_2d = []
    all_opacity = []
    all_color = []

    camera_space_means = []

    for gaussian_idx in range(gaussians.points.shape[0]):
        r = gaussians.quaternions[gaussian_idx : gaussian_idx + 1]
        s = gaussians.scales[gaussian_idx : gaussian_idx + 1]

        hom_points = gaussians.homogeneous_points[gaussian_idx : gaussian_idx + 1]
        j_w_matrix, camera_space_mean = compute_j_w_matrix(camera, hom_points, return_camera_space=True)
        # camera_space_mean = mean_3d_to_camera_space.apply(
        #     hom_points,
        #     camera.extrinsic_matrix,
        # )
        camera_space_mean.retain_grad()
        camera_space_means.append(camera_space_mean)

        pixel_space_mean = camera_space_to_pixel_space.apply(
            camera_space_mean, camera.intrinsic_matrix
        )
        new_pixel_space_mean = torch.zeros_like(pixel_space_mean)
        new_pixel_space_mean[:, :2] = pixel_space_mean[:, :2] / pixel_space_mean[
            :, 3
        ].unsqueeze(1)
        new_pixel_space_mean[:, 2] = pixel_space_mean[:, 2]
        pixel_mean = ndc_to_pixels.apply(new_pixel_space_mean, [height, width])
        final_mean_2d = torch.cat(
            [pixel_mean[:, :2], pixel_mean[:, 2].unsqueeze(1)], dim=1
        )

        color = gaussians.colors[gaussian_idx : gaussian_idx + 1]
        opacity = gaussians.opacity[gaussian_idx : gaussian_idx + 1]
        covariance_2d = r_s_to_cov_2d(r, s, j_w_matrix)
        all_final_means_2d.append(final_mean_2d)
        all_r_s_to_cov_2d.append(covariance_2d)
        all_opacity.append(opacity)
        all_color.append(color)

    target_pixel_x = 17
    target_pixel_y = 16
    for i in range(height):
        for j in range(width):
            if (i != target_pixel_x or j != target_pixel_y) and verbose:
                continue
            current_t = torch.tensor(1.0)
            for gaussian_index in range(gaussians.points.shape[0]):
                color, current_t = render_pixel_custom(
                    torch.tensor([i, j], device=device),
                    all_final_means_2d[gaussian_index][:, :2],
                    all_r_s_to_cov_2d[gaussian_index],
                    all_opacity[gaussian_index],
                    all_color[gaussian_index],
                    current_t,
                )
                image[j, i] += color[0]
    return image

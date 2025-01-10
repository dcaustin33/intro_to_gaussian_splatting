import torch

from splat.test.checked_covariance_derivatives import r_s_to_cov_2d
from splat.test.create_image_auto import (
    Camera,
    Gaussian,
    Gaussian_Covariance_Test,
    compute_2d_covariance,
    compute_j_w_matrix,
)
from splat.test.derivatives import gaussianMeanToPixels, pixelCoordToColor


def create_image(camera: Camera, gaussian: Gaussian, height: int, width: int):
    point_ndc = gaussianMeanToPixels.apply(
        gaussian.mean, 
        torch.tensor(camera.width), 
        torch.tensor(camera.height),
        camera.intrinsic_matrix,
        camera.extrinsic_matrix,
    )
    
    covariance_2d = compute_2d_covariance(
        points_homogeneous=gaussian.homogeneous_points,
        extrinsic_matrix=camera.extrinsic_matrix,
        covariance_3d=gaussian.covariance,
        tan_fovY=camera.tan_fovY,
        tan_fovX=camera.tan_fovX,
        focal_x=camera.focal_x,
        focal_y=camera.focal_y,
    )[0]
    # point_ndc = torch.tensor(point_ndc, requires_grad=True)
    inverted_covariance_2d = torch.linalg.inv(covariance_2d)
    image = torch.zeros((height, width, 3))
    print("point_ndc", point_ndc)
    print("gaussian.color", gaussian.color)
    print("gaussian.opacity", gaussian.opacity)
    for i in range(height):
        for j in range(width):
            image[i, j] = pixelCoordToColor.apply(
                torch.tensor([[i, j]]),
                point_ndc,
                gaussian.color,
                inverted_covariance_2d,
                torch.tensor(1.0),
                gaussian.opacity,
            )
    return image, point_ndc


def create_image_covariance_test(camera: Camera, gaussian: Gaussian_Covariance_Test, height: int, width: int):
    """
    Camera gives the internal and the external matrices.
    Gaussian_Covariance_Test gives the mean_2d, r, s, color, opacity
        Note the mean here has a third dimension for the depth needed for J
    height and width are the dimensions of the image
    """
    image = torch.zeros((height, width, 3))
    r = gaussian.r
    s = gaussian.s
    mean_2d = gaussian.mean_2d
    color = gaussian.color
    opacity = gaussian.opacity

    j_w_matrix = compute_j_w_matrix(camera, mean_2d)
    covariance_2d = r_s_to_cov_2d(r, s, j_w_matrix)
    for i in range(height):
        for j in range(width):
            image[i, j] = pixelCoordToColor.apply(
                torch.tensor([[i, j]]),
                mean_2d[:, :2],
                color,
                covariance_2d,
                torch.tensor(1.0),
                opacity,
            )

    return image
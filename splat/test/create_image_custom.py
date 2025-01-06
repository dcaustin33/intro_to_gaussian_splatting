import torch

from splat.test.create_image_cpu import Camera, Gaussian, compute_2d_covariance
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
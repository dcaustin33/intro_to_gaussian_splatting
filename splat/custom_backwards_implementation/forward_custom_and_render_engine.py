import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from PIL import Image

from splat.custom_pytorch_implementation.create_image_custom import (
    create_image_full_custom_multiple_gaussians_with_splat_gaussians,
)
from splat.gaussians import Gaussians

from splat.custom_pytorch_implementation.create_image_auto import (
    Camera,
    Gaussian,
    create_image_covariance_test_auto,
    create_image_full_auto,
    create_image_full_auto_multiple_gaussians,
    create_image_full_auto_multiple_gaussians_with_splat_gaussians,
)


def return_gaussians() -> Gaussians:
    point_3d1 = torch.tensor([[0.101, 0.1001, -4]], dtype=torch.float64).requires_grad_(
        True
    )
    r1 = torch.tensor([[0.5, 0.02, 0.03, 0.001]], dtype=torch.float64).requires_grad_(
        True
    )
    s1 = torch.tensor([[0.1, 0.15, 0.2]], dtype=torch.float64).requires_grad_(True)
    color1 = torch.tensor([0.4, 0.6, 0.8], dtype=torch.float64).requires_grad_(True)
    opacity1 = torch.tensor([0.5], dtype=torch.float64).requires_grad_(True)

    point_3d2 = torch.tensor(
        [[0.0999, 0.0999, -4.1]], dtype=torch.float64
    ).requires_grad_(True)
    r2 = torch.tensor([[0.2, 0.04, 0.03, 0.001]], dtype=torch.float64).requires_grad_(
        True
    )
    s2 = torch.tensor([[0.2, 0.16, 0.1]], dtype=torch.float64).requires_grad_(True)
    color2 = torch.tensor([0.1, 0.15, 0.2], dtype=torch.float64).requires_grad_(True)
    opacity2 = torch.tensor([0.9], dtype=torch.float64).requires_grad_(True)

    return Gaussians(
        points=torch.stack([point_3d1, point_3d2]),
        colors=torch.stack([color1, color2]),
        scales=torch.stack([s1, s2]),
        quaternions=torch.stack([r1, r2]),
        opacity=torch.stack([opacity1, opacity2]),
    )


def return_camera() -> Camera:
    focal_x = torch.tensor([100.0])
    focal_y = torch.tensor([100.0])
    width = 32
    height = 32
    camera_rotation = torch.tensor([1, 0, 0, 0]).unsqueeze(0)
    camera_translation = torch.tensor([[-0.1, -0.1, 0.0]])

    return Camera(
        focal_x=focal_x,
        focal_y=focal_y,
        c_x=0.0,
        c_y=0.0,
        width=width,
        height=height,
        camera_rotation=camera_rotation,
        camera_translation=camera_translation,
    )


def return_gt_image() -> torch.Tensor:
    gt_image = Image.open("gt.png")
    gt_image = np.array(gt_image) / 255.0
    return torch.tensor(gt_image)


if __name__ == "__main__":
    camera = return_camera()
    gaussians = return_gaussians()
    gt_image = return_gt_image()

    output_auto1 = create_image_full_auto_multiple_gaussians_with_splat_gaussians(
        camera, gaussians, camera.height, camera.width
    )
    output_custom = create_image_full_custom_multiple_gaussians_with_splat_gaussians(
        camera, gaussians, camera.height, camera.width
    )
    print(torch.allclose(output_auto1, output_custom))

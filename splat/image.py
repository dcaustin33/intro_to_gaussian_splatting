from typing import Dict, Tuple

import torch

from splat.read_colmap import Camera, Image
from splat.utils import (
    build_rotation,
    focal2fov,
    get_extrinsic_matrix,
    get_intrinsic_matrix,
    getProjectionMatrix,
    getWorld2View,
    getWorld2View2,
    in_view_frustum,
    ndc2Pix,
)


class GaussianImage(torch.nn.Module):
    def __init__(self, camera: Camera, image: Image) -> None:
        super().__init__()

        self.R = build_rotation(torch.Tensor(image.qvec).unsqueeze(0))
        self.T = torch.Tensor(image.tvec)
        self.height = torch.Tensor([camera.height])
        self.width = torch.Tensor([camera.width])
        self.extrinsic_matrix = get_extrinsic_matrix(self.R[0], self.T)
        self.f_x = torch.Tensor([camera.params[0]])
        self.f_y = torch.Tensor([camera.params[1]])
        self.c_x = torch.Tensor([camera.params[2]])
        self.c_y = torch.Tensor([camera.params[3]])
        self.fovX = focal2fov(self.f_x, self.width)
        self.fovY = focal2fov(self.f_y, self.height)
        self.tan_fovX = torch.tan(self.fovX / 2)
        self.tan_fovY = torch.tan(self.fovY / 2)
        self.intrinsic_matrix = get_intrinsic_matrix(
            f_x=self.f_x, f_y=self.f_y, c_x=self.c_x, c_y=self.c_y
        )

        self.zfar = torch.Tensor([100.0])
        self.znear = torch.Tensor([0.001])
        self.name = image.name

        # this is stolen from the original repo - not sure why the transpose here
        self.world2view = getWorld2View(R=self.R[0], t=self.T).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.fovX, fovY=self.fovY
        ).transpose(0, 1)
        self.full_proj_transform = (
            self.world2view.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world2view.inverse()[3, :3]

        self.projection = self.intrinsic_matrix @ self.extrinsic_matrix

    def project_point_to_camera_perspective_projection(
        self, points: torch.Tensor, colors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # first check if the points will even be in view
        in_frustum_truth = in_view_frustum(
            points=points,
            view_matrix=self.world2view,
        )
        points = points[in_frustum_truth]
        # points will be a nx3 tensor - will make homogenous nx4 tensor
        points = torch.cat(
            [points, torch.ones(points.shape[0], 1, device=points.device)], dim=1
        )
        four_dim_points = points @ self.full_proj_transform  # nx4
        three_dim_points = four_dim_points[:, :3] / four_dim_points[:, 3].unsqueeze(1)
        three_dim_points[:, 0] = ndc2Pix(three_dim_points[:, 0], self.width)
        three_dim_points[:, 1] = ndc2Pix(three_dim_points[:, 1], self.height)
        return three_dim_points, colors[in_frustum_truth]

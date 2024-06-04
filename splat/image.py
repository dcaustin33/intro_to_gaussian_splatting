from typing import Tuple

import torch

from splat.read_colmap import Camera, Image
from splat.utils import (
    build_rotation,
    focal2fov,
    get_extrinsic_matrix,
    get_intrinsic_matrix,
    getProjectionMatrix,
    getWorld2View,
    in_view_frustum,
    ndc2Pix,
)


class GaussianImage(torch.nn.Module):
    def __init__(self, camera: Camera, image: Image) -> None:
        """
        Takes in the camera paramters and the image parameters and creates a
        GaussianImage object that can be used to project points onto the image plane
        """

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.f_x = torch.Tensor([camera.params[0]]).to(self.device)
        self.f_y = torch.Tensor([camera.params[1]]).to(self.device)
        self.c_x = torch.Tensor([camera.params[2]]).to(self.device)
        self.c_y = torch.Tensor([camera.params[3]]).to(self.device)
        self.intrinsic_matrix = get_intrinsic_matrix(
            f_x=self.f_x, f_y=self.f_y, c_x=self.c_x, c_y=self.c_y
        ).to(self.device)
        self.R = build_rotation(torch.Tensor(image.qvec).unsqueeze(0)).to(self.device)
        self.T = torch.Tensor(image.tvec).to(self.device)
        self.height = torch.Tensor([camera.height]).to(self.device)
        self.width = torch.Tensor([camera.width]).to(self.device)
        self.extrinsic_matrix = get_extrinsic_matrix(self.R[0], self.T).to(self.device)
        self.fovX = focal2fov(self.f_x, self.width).to(self.device)
        self.fovY = focal2fov(self.f_y, self.height).to(self.device)
        self.tan_fovX = torch.tan(self.fovX / 2).to(self.device)
        self.tan_fovY = torch.tan(self.fovY / 2).to(self.device)

        # arbitrary selection that the authors make
        self.zfar = torch.Tensor([100.0]).to(self.device)
        self.znear = torch.Tensor([0.001]).to(self.device)
        self.name = image.name

        # this is stolen from the original repo - not sure why the transpose here
        self.world2view = (
            getWorld2View(R=self.R[0], t=self.T).transpose(0, 1).to(self.device)
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.fovX, fovY=self.fovY
            )
            .transpose(0, 1)
            .to(self.device)
        )
        self.full_proj_transform = (
            (self.world2view.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)))
            .squeeze(0)
            .to(self.device)
        )
        self.camera_center = self.world2view.inverse()[3, :3].to(self.device)

        self.projection = (self.intrinsic_matrix @ self.extrinsic_matrix).to(
            self.device
        )

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
            [points, torch.ones(points.shape[0], 1, device=self.device)], dim=1
        )
        four_dim_points = points @ self.full_proj_transform  # nx4
        three_dim_points = four_dim_points[:, :3] / four_dim_points[:, 3].unsqueeze(1)
        three_dim_points[:, 0] = ndc2Pix(three_dim_points[:, 0], self.width)
        three_dim_points[:, 1] = ndc2Pix(three_dim_points[:, 1], self.height)
        return three_dim_points, colors[in_frustum_truth]

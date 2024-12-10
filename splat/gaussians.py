from typing import Optional

import torch
from torch import nn

from splat.utils import build_rotation, inverse_sigmoid


class Gaussians(nn.Module):
    def __init__(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        quaternions: Optional[torch.Tensor] = None,
        opacity: Optional[torch.Tensor] = None,
        requires_grad: bool = False,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.points = points.clone().requires_grad_(requires_grad).to(self.device).float()
        if colors.max() > 1:
            divisor = 256
        else:
            divisor = 1
        self.colors = (
            (colors / divisor).clone().requires_grad_(requires_grad).to(self.device).float()
        )
        if scales is None:
            self.scales = torch.ones((len(self.points), 3)).to(self.device).float() * 0.005
        else:
            self.scales = scales.clone().requires_grad_(requires_grad).to(self.device).float()

        # for now we do not use initialize scale
        # however from the paper the scale is initialized using
        # mean of the three smallest nonzero distances for each point
        # self.initialize_scale()
        
        if quaternions is None:
            self.quaternions = torch.zeros((len(self.points), 4)).to(self.device)
            self.quaternions[:, 0] = 1.0
        else:
            self.quaternions = quaternions.clone().requires_grad_(requires_grad).to(self.device)

        if opacity is None:
            self.opacity = inverse_sigmoid(
                0.9999 * torch.ones((self.points.shape[0], 1), dtype=torch.float)
            ).to(self.device)
        else:
            self.opacity = opacity.clone().requires_grad_(requires_grad).to(self.device)

    def initialize_scale(
        self,
    ) -> None:
        """Finds the third closest neighbor and uses it as the scale"""
        point_diffs = self.points.unsqueeze(0) - self.points.unsqueeze(1)
        distances = torch.linalg.norm(point_diffs, dim=2)

        # Set diagonal to a large number to ignore zero distance to itself
        distances.fill_diagonal_(float("inf"))

        # Sort distances and take the mean of the three smallest nonzero distances for each point
        closest_distances = distances.sort(dim=1).values[:, :3]
        all_scale = closest_distances.mean(dim=1)
        all_scale = torch.clamp(all_scale, min=0.00001)

        # Update scales
        # self.scales *= torch.log(torch.sqrt(all_scale)).unsqueeze(1)
        self.scales *= all_scale.unsqueeze(1)

    def get_3d_covariance_matrix(self) -> torch.Tensor:
        """
        Get the 3D covariance matrix from the scale and rotation matrix
        """
        # noramlize the quaternions
        quaternions = nn.functional.normalize(self.quaternions, p=2, dim=1)
        # nx3x3 matrix
        rotation_matrices = build_rotation(quaternions)
        # nx3x3 matrix
        scale_matrices = torch.zeros((len(self.points), 3, 3)).to(self.device)
        scale_matrices[:, 0, 0] = self.scales[:, 0]
        scale_matrices[:, 1, 1] = self.scales[:, 1]
        scale_matrices[:, 2, 2] = self.scales[:, 2]
        scale_rotation_matrix = rotation_matrices @ scale_matrices
        covariance = scale_rotation_matrix @ scale_rotation_matrix.transpose(1, 2)
        return covariance

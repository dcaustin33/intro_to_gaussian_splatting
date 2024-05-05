from torch import nn
import torch

from splat.read_colmap import qvec2rotmat, qvec2rotmat_matrix


class GaussianScene(nn.Module):
    def __init__(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        e_opacity: float = 0.005,
        divide_scale: float = 1.6,
        gradient_pos_threshold: float = 0.0002,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # nx3 matrix
        self.points = torch.tensor(
            points, dtype=torch.float32, requires_grad=True, device=self.device
        )
        # nx3 matrix
        self.colors = torch.tensor(
            colors, dtype=torch.float32, requires_grad=True, device=self.device
        )
        # nx1 matrix
        self.opacity = torch.tensor(
            [0.5] * len(points),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        # nx4 matrix
        self.quaternions = torch.tensor(
            [[0, 0, 0, 1]] * len(points),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        # nx3 matrix
        self.scales = torch.tensor(
            [[1, 1, 1]] * len(points),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        # jitter to break symmetry
        with torch.no_grad():
            self.quaternions += (
                torch.randn(len(points), 4, device=self.device) * 0.000001
            )
            self.scales += torch.randn(len(points), 3, device=self.device) * 0.000001

        # used for densifying and removing gaussians
        self.e_opacity = e_opacity
        self.divide_scale = divide_scale
        # this corresponds to densify_grad_threshold in the original code
        self.gradient_pos_threshold = gradient_pos_threshold
        self.size_threshold = 20

    def get_3d_covariance_matrix(self) -> torch.Tensor:
        """
        Get the 3D covariance matrix from the scale and rotation matrix
        """
        # nx3x3 matrix
        rotation_matrices = torch.stack([qvec2rotmat(q) for q in self.quaternions])
        # nx3x3 matrix
        scale_matrices = torch.stack([torch.diag(s) for s in self.scales])
        scale_rotation_matrix = rotation_matrices @ scale_matrices
        covariance = scale_rotation_matrix @ scale_rotation_matrix.transpose(1, 2)
        return covariance

    def remove_gaussian(
        self,
    ) -> None:
        """Removes the gaussians that are essentially transparent"""
        with torch.no_grad():
            opacities = torch.sigmoid(self.opacity)
            truth = opacities > self.e_opacity
            self.points = self.points[truth]
            self.colors = self.colors[truth]
            self.opacity = self.opacity[truth]
            self.quaternions = self.quaternions[truth]
            self.scales = self.scales[truth]

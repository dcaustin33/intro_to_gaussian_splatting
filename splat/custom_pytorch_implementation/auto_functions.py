"""This file has equivalently named functions that use autograd from checked_covariance_derivatives.py"""

import torch

from splat.utils import build_rotation


def invert_2x2_matrix_auto(matrix: torch.Tensor):
        """input is a nx2x2 tensor, returns the inverse of each 2x2 matrix"""
        det = matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
        # Create empty nx2x2 tensor for the inverted matrices
        final_matrices = torch.zeros_like(matrix)

        # Fill in the inverted matrix elements using the 2x2 matrix inverse formula
        final_matrices[:, 0, 0] = matrix[:, 1, 1] / det
        final_matrices[:, 0, 1] = -matrix[:, 0, 1] / det
        final_matrices[:, 1, 0] = -matrix[:, 1, 0] / det
        final_matrices[:, 1, 1] = matrix[:, 0, 0] / det

        if torch.isinf(final_matrices).any():
            raise RuntimeError("Infinite values in final matrices")
        return final_matrices

def covariance_3d_to_covariance_2d_auto(covariance_3d: torch.Tensor, U: torch.Tensor):
    """
    Covariance 3d is the nx3x3 covariance matrix.
    U is the J@W.T matrix. this is a 3x3 matrix
    To get the covariance 2d we do U.T @ covariance_3d @ U
    """
    if U.shape[0] == 1:
         U = U.squeeze(0)
    first_mult = torch.einsum("ij,njk->nik", U.transpose(0, 1), covariance_3d)
    # second_mult = first_mult @ U
    second_mult = torch.einsum("nij,jk->nik", first_mult, U)
    return second_mult

def R_S_To_M_auto(R: torch.Tensor, S: torch.Tensor):
    """R is a nx3x3 rotation matrix, S is a nx3x3 scale matrix"""
    return torch.einsum("nij,njk->nik", R, S)

def M_To_Covariance_auto(M: torch.Tensor):
    return M.pow(2)

def quats_to_R_auto(quats: torch.Tensor) -> torch.Tensor:
    R = build_rotation(quats, normalize=False)
    return R

def normalize_quats_auto(quats: torch.Tensor):
    """Quats are a nx4 tensor"""
    return quats / quats.norm(dim=1, keepdim=True)

def scale_to_s_matrix_auto(s: torch.Tensor):
    """Takes the nx3 tensor and returns the nx3x3 diagonal matrix"""
    return torch.diag_embed(s)


def r_s_to_cov_2d_auto(r: torch.Tensor, s: torch.Tensor, U: torch.Tensor):
     """
     r is a unnormalized nx4 tensor, s is a nx3 tensor
     U is the J@W matrix. this is a 3x3 matrix.
     In our example its W.t since we are using the OpenGL convention
     """
     r = normalize_quats_auto(r)
     R = quats_to_R_auto(r)
     S = scale_to_s_matrix_auto(s)
     M = R_S_To_M_auto(R, S)
     cov_3d = M_To_Covariance_auto(M)
     cov_2d = covariance_3d_to_covariance_2d_auto(cov_3d, U)
     inv_cov_2d = invert_2x2_matrix_auto(cov_2d[:, :2, :2])
     return inv_cov_2d

def gaussian_weight_auto(gaussian_mean: torch.Tensor, inverted_covariance: torch.Tensor, pixel: torch.Tensor):
    """
    gaussian_mean: nx2 tensor
    inverted_covariance: nx2x2 tensor
    pixel: 1x2 tensor
    """
    diff = (pixel - gaussian_mean).unsqueeze(1)
    # 2x2 * 2x1 = 2x1
    inv_cov_mult = torch.einsum('bij,bjk->bik', inverted_covariance, diff.transpose(1, 2))
    return -0.5 * torch.einsum('bij,bjk->bik', diff, inv_cov_mult).squeeze(-1)

def gaussian_exp_auto(gaussian_weight: torch.Tensor):
    return torch.exp(gaussian_weight)

def get_alpha_auto(gaussian_strength: torch.Tensor, unactivated_opacity: torch.Tensor):
    return gaussian_strength * torch.sigmoid(unactivated_opacity)

def final_color_auto(color: torch.Tensor, current_T: torch.Tensor, alpha: torch.Tensor):
    test_t = current_T * (1 - alpha)
    return color * current_T * alpha, test_t

def render_pixel_auto(
    pixel_value: torch.Tensor,
    mean_2d: torch.Tensor,
    inv_covariance_2d: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    current_T: torch.Tensor,
):
    g_weight = gaussian_weight_auto(mean_2d, inv_covariance_2d, pixel_value)
    g_strength = gaussian_exp_auto(g_weight)
    alpha = get_alpha_auto(g_strength, opacity)
    final_color, current_T = final_color_auto(color, current_T, alpha)
    return final_color, torch.tensor(current_T)

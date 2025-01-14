import torch
from torch.autograd.gradcheck import gradcheck

from splat.utils import build_rotation


class final_color(torch.autograd.Function):
    @staticmethod
    def forward(ctx, color: torch.Tensor, current_T: torch.Tensor, alpha: torch.Tensor):
        """Color is a nx3 tensor, weight is a nx1 tensor, alpha is a nx1 tensor"""
        ctx.save_for_backward(color, current_T, alpha)
        return color * current_T * alpha
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Output of forward is a nx3 tensor so the grad_output is a nx3 tensor"""
        color, current_T, alpha = ctx.saved_tensors
        grad_color = grad_output * current_T * alpha
        grad_alpha = (grad_output * color * current_T).sum(dim=1, keepdim=True)
        return grad_color, None, grad_alpha
    
class get_alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussian_strength: torch.Tensor, unactivated_opacity: torch.Tensor):
        """Gaussian strength is a nx1 tensor, unactivated opacity is a nx1 tensor"""
        ctx.save_for_backward(gaussian_strength, unactivated_opacity)
        activated_opacity = torch.sigmoid(unactivated_opacity)
        return gaussian_strength * activated_opacity
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Output of forward is a nx1 tensor so the grad_output is a nx1 tensor"""
        gaussian_strength, unactivated_opacity = ctx.saved_tensors
        derivative_sigmoid = torch.sigmoid(unactivated_opacity) * (1 - torch.sigmoid(unactivated_opacity))
        grad_gaussian_strength = grad_output * torch.sigmoid(unactivated_opacity)
        grad_unactivated_opacity = grad_output * gaussian_strength * derivative_sigmoid
        return grad_gaussian_strength, grad_unactivated_opacity


class gaussian_exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussian_weight: torch.Tensor):
        ctx.save_for_backward(gaussian_weight)
        return torch.exp(gaussian_weight)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gaussian_weight = ctx.saved_tensors[0]
        grad_gaussian_weight = grad_output * torch.exp(gaussian_weight)
        return grad_gaussian_weight
    
class gaussian_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussian_mean: torch.Tensor, inverted_covariance: torch.Tensor, pixel: torch.Tensor):
        """
        Pixel means are a nx2 tensor, inverted covariance is a 2x2 tensor, pixel is a nx2 tensor
        Outputs a nx1 tensor
        """
        ctx.save_for_backward(gaussian_mean, inverted_covariance, pixel)
        diff = (pixel - gaussian_mean).unsqueeze(1)
        # 2x2 * 2x1 = 2x1
        inv_cov_mult = torch.einsum('bij,bjk->bik', inverted_covariance, diff.transpose(1, 2))
        return -0.5 * torch.einsum('bij,bjk->bik', diff, inv_cov_mult).squeeze(-1)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Output of forward is a nx1 tensor so the grad_output is a nx1 tensor"""
        gaussian_mean, inverted_covariance, pixel = ctx.saved_tensors
        diff = (pixel - gaussian_mean).unsqueeze(1)  # nx2x1

        deriv_wrt_inv_cov = -0.5 * torch.einsum("bij,bjk->bik", diff.transpose(1, 2), diff)
        grad_inv_cov = grad_output * deriv_wrt_inv_cov  # output is nx2x2

        deriv_wrt_diff = -0.5 * 2 * torch.einsum("bij,bjk->bik", diff, inverted_covariance)
        deriv_wrt_gaussian_mean = -1
        grad_gaussian_mean = torch.einsum("bi,bij->bj", grad_output, deriv_wrt_diff) * deriv_wrt_gaussian_mean
        return grad_gaussian_mean, grad_inv_cov, None
    
def render_pixel_custom(
        pixel_value: torch.Tensor,
        gaussian_mean: torch.Tensor,
        inverted_covariance: torch.Tensor,
        opacity: torch.Tensor,
        color: torch.Tensor,
        current_T: torch.Tensor,
) -> torch.Tensor:
    """
    Inputs:
        gaussian_mean: nx2 tensor
        inverted_covariance: nx2x2 tensor
        opacity: nx1 tensor
        color: nx3 tensor
        current_T: nx1 tensor
        pixel_value: 1x2 tensor
    """
    g_weight = gaussian_weight.apply(gaussian_mean, inverted_covariance, pixel_value)
    g_strength = gaussian_exp.apply(g_weight)
    alpha = get_alpha.apply(g_strength, opacity)
    color_output = final_color.apply(color, current_T, alpha)
    return color_output
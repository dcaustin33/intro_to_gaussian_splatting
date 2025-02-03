import torch
from torch.autograd.gradcheck import gradcheck

from splat.custom_pytorch_implementation.auto_functions import gaussian_weight_auto
from splat.utils import build_rotation


class final_color(torch.autograd.Function):
    @staticmethod
    def forward(ctx, color: torch.Tensor, current_T: torch.Tensor, alpha: torch.Tensor):
        """Color is a nx3 tensor, weight is a nx1 tensor, alpha is a nx1 tensor"""
        ctx.save_for_backward(color, current_T, alpha)
        test_t = current_T * (1 - alpha)
        if alpha < 1.0 / 255.0:
            return torch.zeros_like(color), current_T
        return color * current_T * alpha, test_t
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_test_t: torch.Tensor):
        """Output of forward is a nx3 tensor so the grad_output is a nx3 tensor"""
        color, current_T, alpha = ctx.saved_tensors
        if alpha < 1.0 / 255.0:
            return torch.zeros_like(color), None, torch.zeros_like(alpha)
        grad_color = grad_output * current_T * alpha
        grad_alpha = (grad_output * color * current_T).sum(dim=1, keepdim=True)
        # if grad_output.sum() != 0:
        #     print("grad_output in custom:", grad_output)
        #     print("alpha:", alpha, "current_T:", current_T)
        #     print("grad_alpha:", grad_alpha)
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
        # if grad_output.sum() != 0:
        #     print("grad_output:", grad_output)
        #     print("grad_gaussian_strength:", grad_gaussian_strength.item())
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
        # if grad_output.sum() != 0:
        #     print("grad_gaussian_weight:", grad_gaussian_weight.item())
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

        inv_cov_mult = torch.einsum('bij,bjk->bik', inverted_covariance, diff.transpose(1, 2))
        return -0.5 * torch.einsum('bij,bjk->bik', diff, inv_cov_mult).squeeze(-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Output of forward is a nx1 tensor so the grad_output is a nx1 tensor"""
        gaussian_mean, inverted_covariance, pixel = ctx.saved_tensors
        diff = (pixel - gaussian_mean).unsqueeze(1)  # nx1x2

        deriv_wrt_inv_cov = -0.5 * torch.einsum("bij,bjk->bik", diff.transpose(1, 2), diff)
        grad_inv_cov = torch.einsum("bi,bjk->bjk", grad_output, deriv_wrt_inv_cov)

        deriv_output_wrt_diff1 = torch.einsum("bij,bjk->bik", inverted_covariance, diff.transpose(1, 2))
        deriv_output_wrt_diff2 = torch.einsum("bij,bjk->bik", inverted_covariance.transpose(1, 2), diff.transpose(1, 2))

        deriv_output_wrt_diff = -0.5 * torch.einsum("bi,bji->bj", grad_output, deriv_output_wrt_diff1 + deriv_output_wrt_diff2)
        grad_gaussian_mean = deriv_output_wrt_diff * -1
        # if grad_output.sum() != 0:
        #     print("inverted_covariance:", inverted_covariance)
        #     print("deriv_wrt_inv_cov:", deriv_wrt_inv_cov)
        #     print("grad_inv_cov:", grad_inv_cov)
        #     print("\n\n")
        #     print("diff:", diff)
        #     print("deriv_output_wrt_diff1:", deriv_output_wrt_diff1)
        #     print("deriv_output_wrt_diff2:", deriv_output_wrt_diff2)
        #     print("deriv_output_wrt_diff:", deriv_output_wrt_diff)
        #     print("grad_output:", grad_output)
        #     print("grad_gaussian_mean:", grad_gaussian_mean[0, 0].item(), grad_gaussian_mean[0, 1].item())

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
    color_output, current_T = final_color.apply(color, current_T, alpha)
    return color_output, current_T


class mean_3d_to_camera_space(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean_3d: torch.Tensor, extrinsic_matrix: torch.Tensor):
        ctx.save_for_backward(extrinsic_matrix)
        return torch.einsum("nk, kh->nh", mean_3d, extrinsic_matrix)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        extrinsic_matrix = ctx.saved_tensors[0]
        mean_3d_grad = torch.einsum("nh,hj->nj", grad_output, extrinsic_matrix.transpose(0, 1))
        return mean_3d_grad, None
    

class camera_space_to_pixel_space(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean_3d: torch.Tensor, intrinsic_matrix: torch.Tensor):
        ctx.save_for_backward(intrinsic_matrix)
        return torch.einsum("nk, kh->nh", mean_3d, intrinsic_matrix)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):   
        intrinsic_matrix = ctx.saved_tensors[0]
        mean_3d_grad = torch.einsum("nh,hj->nj", grad_output, intrinsic_matrix.transpose(0, 1))
        return mean_3d_grad, None
    
class ndc_to_pixels(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ndc: torch.Tensor, dimension: list):
        """ndc is a nx3 tensor where the last dimension is the z component
        
        dimension are the height and width of the image
        """
        ctx.save_for_backward(torch.tensor(dimension))
        ndc = ndc.clone()  # To avoid modifying input in-place
        ndc[:, 0] = (ndc[:, 0] + 1) * (dimension[1] - 1) * 0.5
        ndc[:, 1] = (ndc[:, 1] + 1) * (dimension[0] - 1) * 0.5
        return ndc
    
    @staticmethod
    def backward(ctx, grad_output):
        dimension = ctx.saved_tensors[0]
        grad_ndc = grad_output.clone()

        # Compute the gradient for ndc
        grad_ndc[:, 0] *= (dimension[1] - 1) * 0.5
        grad_ndc[:, 1] *= (dimension[0] - 1) * 0.5
        # grad_ndc[:, 2] = 0  # z-component has no effect on pixel coordinates
        return grad_ndc, None
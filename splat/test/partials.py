import torch

from splat.utils import ndc2Pix


def d_color_wrt_alpha(C: torch.Tensor, T_at_the_time: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of color wrt alpha

    C should be a nx3 tensor
    T_at_the_time should be a nx1 tensor

    Output should be a nx3 tensor
    """
    return C * T_at_the_time


def d_alpha_wrt_gaussian_strength(opacity: torch.Tensor) -> torch.Tensor:
    """
    input is a nx1 tensor
    output is a nx1 tensor
    """
    return torch.sigmoid(opacity)


def d_gaussian_strength_wrt_gaussian_weight(
    gaussian_weight: torch.Tensor,
) -> torch.Tensor:
    """
    input is a nx1 tensor
    output is a nx1 tensor
    """
    return torch.exp(gaussian_weight).view(-1, 1)


def d_gaussian_weight_wrt_diff(
    diff: torch.Tensor, inv_covariance: torch.Tensor
) -> torch.Tensor:
    """
    Compute the derivative of gaussian weight wrt diff.
    Derivative of x.T @ inv_covariance @ x is (inv_covariance + inv_covariance.T) @ x

    diff should be a nx2 tensor
    inv_covariance should be a nx2x2 tensor

    output should be a n,2 tensor
    """
    return -0.5 * torch.bmm(inv_covariance + inv_covariance, diff.transpose(1, 2))


def d_diff_wrt_mean(n: int) -> torch.Tensor:
    """n, 2, 2 tensor of -1 on diagonal"""
    return -1 * torch.eye(2).unsqueeze(0).repeat(n, 1, 1)


def d_pixel_space_to_pixels(width: int, height: int) -> torch.Tensor:
    """
    Compute the derivative of pixel space to pixels
    """
    return torch.tensor([[(width - 1) / 2, 0], [0, (height - 1) / 2]])


def d_pixel_space_perspective_transform(
    x_value_before_divide: torch.Tensor,
    y_value_before_divide: torch.Tensor,
    w_value_before_divide: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the derivative of pixel space perspective transform
    """
    deriviative = torch.zeros((x_value_before_divide.shape[0], 2, 4))
    deriviative[:, 0, 0] = 1 / w_value_before_divide
    deriviative[:, 1, 1] = 1 / w_value_before_divide
    deriviative[:, 0, 3] = -x_value_before_divide / (w_value_before_divide**2)
    deriviative[:, 1, 3] = -y_value_before_divide / (w_value_before_divide**2)
    return deriviative


def d_camera_to_pixel_space(intrinsic_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of points homogeneous @ intrinsic_matrix
    """
    return intrinsic_matrix.T


def d_world_to_camera(extrinsic_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of points homogeneous @ extrinsic_matrix
    """
    return extrinsic_matrix.T


def extract_gaussian_weight(
    pixel: torch.Tensor,
    mean: torch.Tensor,
    inv_covariance: torch.Tensor,
    pdb: bool = False,
) -> torch.Tensor:
    """
    Use the covariance matrix to extract the weight of the point

    Args:
        mean: 1x2 tensor
        covariance: 2x2 tensor
    """
    diff = pixel - mean
    diff = diff.unsqueeze(0)
    gaussian_weight = -0.5 * torch.matmul(diff, torch.matmul(inv_covariance, diff.t()))
    actual_weight = torch.exp(gaussian_weight)
    return actual_weight, gaussian_weight


def render_pixel(
    x_value: int,
    y_value: int,
    mean_2d: torch.Tensor,
    inv_covariance_2d: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    current_T: float,
    min_weight: float = 0.00001,
    verbose: bool = False,
):
    """Uses alpha blending to render a pixel"""
    gaussian_strength, exponent_weight = extract_gaussian_weight(
        torch.Tensor([x_value, y_value]), mean_2d, inv_covariance_2d
    )
    alpha = gaussian_strength * torch.sigmoid(opacity)
    test_t = current_T * (1 - alpha)
    if verbose:
        print(
            f"x_value: {x_value}, y_value: {y_value}, gaussian_strength: {gaussian_strength}, alpha: {alpha}, test_t: {test_t}, mean_2d: {mean_2d}"
        )
    if test_t < min_weight:
        return
    return (
        color * current_T * alpha,
        test_t,
        current_T,
        gaussian_strength,
        exponent_weight,
    )


class gaussianMeanToPixels(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points_3d: torch.Tensor,
        width: int,
        height: int,
        intrinsic_matrix: torch.Tensor,
        extrinsic_matrix: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            width, height, intrinsic_matrix, extrinsic_matrix, points_3d
        )
        points_homogeneous = torch.cat(
            [points_3d, torch.ones(points_3d.shape[0], 1)], dim=1
        )
        points_camera_space = points_homogeneous @ extrinsic_matrix
        points_pixel_space_before_divide = points_camera_space @ intrinsic_matrix
        points_pixel_space = points_pixel_space_before_divide[
            :, :2
        ] / points_pixel_space_before_divide[:, 3].unsqueeze(1)
        pixel_x = ndc2Pix(points_pixel_space[:, 0], width)
        pixel_y = ndc2Pix(points_pixel_space[:, 1], height)
        final_coords = torch.cat([pixel_x.view(-1, 1), pixel_y.view(-1, 1)], dim=1)
        return final_coords

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """We are going to kind of do checkpointing and compute the values we need here"""
        width, height, intrinsic_matrix, extrinsic_matrix, points_3d = ctx.saved_tensors
        ndc_before_divide = (
            torch.cat([points_3d, torch.ones(points_3d.shape[0], 1)], dim=1)
            @ extrinsic_matrix
            @ intrinsic_matrix
        )
        derivative = d_pixel_space_to_pixels(width, height)
        derivative = derivative @ d_pixel_space_perspective_transform(ndc_before_divide[:, 0], ndc_before_divide[:, 1], ndc_before_divide[:, 3])
        derivative = derivative @ d_camera_to_pixel_space(intrinsic_matrix)
        derivative = derivative @ d_world_to_camera(extrinsic_matrix)
        gradient = torch.bmm(grad_output.unsqueeze(1), derivative).squeeze(1)
        return gradient[:, :3], None, None, None, None, None

class pixelCoordToColor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pixel_coords: torch.Tensor,
        gaussian_coords: torch.Tensor,
        color: torch.Tensor,
        inv_covariance_2d: torch.Tensor,
        current_T: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        """
        pixel_coords: nx2 tensor
        gaussian_coords: nx2 tensor
        color: nx3 tensor
        inv_covariance_2d: nx2x2 tensor
        current_T: nx1 tensor
        opacity: nx1 tensor
        """
        diff = pixel_coords - gaussian_coords
        diff = diff.unsqueeze(-1).transpose(1, 2)
        gaussian_weight = -0.5 * torch.matmul(
            diff, torch.matmul(inv_covariance_2d, diff.transpose(1, 2))
        )
        actual_weight = torch.exp(gaussian_weight).view(-1, 1)
        alpha = actual_weight * torch.sigmoid(opacity)
        ctx.save_for_backward(
            color, inv_covariance_2d, current_T, opacity, gaussian_weight, diff, alpha, actual_weight
        )
        return color * current_T * alpha

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        color, inv_covariance_2d, current_T, opacity, gaussian_weight, diff, alpha, actual_weight = (
            ctx.saved_tensors
        )
        derivative = d_color_wrt_alpha(color, current_T).view(-1, 3, 1)
        derivative = derivative * d_alpha_wrt_gaussian_strength(opacity).unsqueeze(1)
        derivative = derivative * d_gaussian_strength_wrt_gaussian_weight(
            gaussian_weight
        ).unsqueeze(1)
        derivative = torch.bmm(
            derivative,
            d_gaussian_weight_wrt_diff(diff, inv_covariance_2d).transpose(1, 2),
        )
        derivative = torch.bmm(derivative, d_diff_wrt_mean(diff.shape[0]))
        gradient = torch.bmm(grad_output.unsqueeze(1), derivative).squeeze(1)
        color_gradient = grad_output * (current_T * alpha)
        # import pdb; pdb.set_trace()
        opacity_gradient = (grad_output * (color * current_T)).sum(dim=1, keepdim=True)
        opacity_gradient *= actual_weight * (torch.sigmoid(opacity) * (1 - torch.sigmoid(opacity)))
        return None, gradient, color_gradient, None, None, opacity_gradient.sum(dim=0)
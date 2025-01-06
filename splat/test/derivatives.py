import torch

from splat.utils import ndc2Pix
from splat.utils import build_rotation


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


def d_inv_covariance_wrt_covariance_a(covariance: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of inv_covariance wrt covariance
    covariance is nx2x2
    Output is nx1
    """
    a = covariance[:, 0, 0]
    b = covariance[:, 0, 1]
    c = covariance[:, 1, 1]

    det_squared = (a * c - b**2) ** 2
    first_component = (-(c**2)) / det_squared
    second_component = 2 * ((b * c) / det_squared)
    third_component = (-(b**2)) / det_squared
    return first_component + second_component + third_component


def d_inv_covariance_wrt_covariance_b(covariance: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of inv_covariance wrt covariance
    covariance is nx2x2
    Output is nx1
    """
    a = covariance[:, 0, 0]
    b = covariance[:, 0, 1]
    c = covariance[:, 1, 1]

    det_squared = (a * c - b**2) ** 2
    first_component = (2 * b * c) / det_squared
    second_component = 2 * ((-a * c - b * b) / det_squared)
    third_component = (2 * a * b) / det_squared
    return first_component + second_component + third_component


def d_inv_covariance_wrt_covariance_c(covariance: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of inv_covariance wrt covariance
    covariance is nx2x2
    Output is nx1
    """
    a = covariance[:, 0, 0]
    b = covariance[:, 0, 1]
    c = covariance[:, 1, 1]

    det_squared = (a * c - b**2) ** 2
    first_component = (-b * b) / det_squared
    second_component = 2 * ((a * b) / det_squared)
    third_component = (-a * a) / det_squared
    return first_component + second_component + third_component


def d_inv_covariance_wrt_covariance(covariance: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of inv_covariance wrt covariance
    covariance is nx2x2
    Output is nx2x2
    """
    d_wrt_a = d_inv_covariance_wrt_covariance_a(covariance)
    d_wrt_b = d_inv_covariance_wrt_covariance_b(covariance)
    d_wrt_c = d_inv_covariance_wrt_covariance_c(covariance)
    result = torch.zeros((covariance.shape[0], 2, 2), device=covariance.device)
    result[:, 0, 0] = d_wrt_a
    result[:, 0, 1] = d_wrt_b
    result[:, 1, 0] = d_wrt_b
    result[:, 1, 1] = d_wrt_c
    return result


def d_covariance_2d_wrt_covariance_3d(
    covariance_3d: torch.Tensor, u: torch.Tensor, grad_output: torch.Tensor
) -> torch.Tensor:
    """
    Compute the derivative of covariance_2d wrt covariance_3d
    covariance_3d is nx3x3
    u is nx2x3
    grad_output is nx2x2
    """
    return torch.bmm(u.transpose(1, 2), torch.bmm(grad_output, u))


def d_covariance_3d_wrt_m(m: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of covariance_3d wrt m
    covariance_3d = M @ M.transpose(1, 2)
    where M = R@S, R is rotation, S is scale matrices
    """
    return 2 * m.transpose(1, 2)


def d_r_wrt_qr(quats: torch.Tensor, n: int) -> torch.Tensor:
    """
    Compute the derivative of m wrt quats
    quats is nx4 tensor
    shape is nx3 tensor
    """
    qr = quats[:, 0]
    qi = quats[:, 1]
    qj = quats[:, 2]
    qk = quats[:, 3]

    derivative = torch.zeros((n, 3, 3))
    derivative[:, 0, 1] = -qk
    derivative[:, 0, 2] = qj
    derivative[:, 1, 0] = qk
    derivative[:, 1, 2] = -qi
    derivative[:, 2, 0] = -qj
    derivative[:, 2, 1] = qi

    return 2 * derivative


def d_r_wrt_qi(quats: torch.Tensor, n: int) -> torch.Tensor:
    """
    Compute the derivative of m wrt quats
    quats is nx4 tensor
    shape is nx3 tensor
    """
    qr = quats[:, 0]
    qi = quats[:, 1]
    qj = quats[:, 2]
    qk = quats[:, 3]

    derivative = torch.zeros((n, 3, 3))
    derivative[:, 0, 1] = qj
    derivative[:, 0, 2] = qk
    derivative[:, 1, 0] = qj
    derivative[:, 1, 1] = -2 * qi
    derivative[:, 1, 2] = -qr
    derivative[:, 2, 0] = qk
    derivative[:, 2, 1] = qr
    derivative[:, 2, 2] = -2 * qi
    return 2 * derivative


def d_r_wrt_qj(quats: torch.Tensor, n: int) -> torch.Tensor:
    """
    Compute the derivative of m wrt quats
    quats is nx4 tensor
    shape is nx3 tensor
    """
    qr = quats[:, 0]
    qi = quats[:, 1]
    qj = quats[:, 2]
    qk = quats[:, 3]

    derivative = torch.zeros((n, 3, 3))
    derivative[:, 0, 0] = -2 * qj
    derivative[:, 0, 1] = qi
    derivative[:, 0, 2] = qr
    derivative[:, 1, 0] = qi
    derivative[:, 1, 1] = 0
    derivative[:, 1, 2] = qk
    derivative[:, 2, 0] = -qr
    derivative[:, 2, 1] = qk
    derivative[:, 2, 2] = -2 * qj
    return 2 * derivative


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
        derivative = derivative @ d_pixel_space_perspective_transform(
            ndc_before_divide[:, 0], ndc_before_divide[:, 1], ndc_before_divide[:, 3]
        )
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
            color,
            inv_covariance_2d,
            current_T,
            opacity,
            gaussian_weight,
            diff,
            alpha,
            actual_weight,
        )
        return color * current_T * alpha

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (
            color,
            inv_covariance_2d,
            current_T,
            opacity,
            gaussian_weight,
            diff,
            alpha,
            actual_weight,
        ) = ctx.saved_tensors
        derivative = (
            grad_output * d_color_wrt_alpha(color, current_T).view(-1, 3)
        ).sum(dim=1, keepdim=True)
        derivative = derivative * d_alpha_wrt_gaussian_strength(opacity).unsqueeze(1)
        derivative = derivative * d_gaussian_strength_wrt_gaussian_weight(
            gaussian_weight
        )
        derivative = derivative * d_gaussian_weight_wrt_diff(
            diff, inv_covariance_2d
        ).transpose(1, 2)
        derivative = torch.bmm(derivative, d_diff_wrt_mean(diff.shape[0]))
        # commenting this out for now but summing above does lead to some precision differences with autograd
        # think they are equivalent but z coord grad is a little off
        # gradient = torch.bmm(grad_output.unsqueeze(1), derivative).squeeze(1)
        color_gradient = grad_output * (current_T * alpha)

        opacity_gradient = (grad_output * (color * current_T)).sum(dim=1, keepdim=True)
        opacity_gradient *= actual_weight * (
            torch.sigmoid(opacity) * (1 - torch.sigmoid(opacity))
        )
        return None, derivative, color_gradient, None, None, opacity_gradient.sum(dim=0)


class quatsToR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quats: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(quats)
        R = build_rotation(quats)
        return R

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        quats = ctx.saved_tensors[0]
        deriv_wrt_qr = d_r_wrt_qr(quats, quats.shape[0])
        deriv_wrt_qi = d_r_wrt_qi(quats, quats.shape[0])
        deriv_wrt_qj = d_r_wrt_qj(quats, quats.shape[0])

        deriv_wrt_qr = (
            (grad_output * deriv_wrt_qr).sum(dim=(1, 2), keepdim=True).squeeze(2)
        )
        deriv_wrt_qi = (
            (grad_output * deriv_wrt_qi).sum(dim=(1, 2), keepdim=True).squeeze(2)
        )
        deriv_wrt_qj = (
            (grad_output * deriv_wrt_qj).sum(dim=(1, 2), keepdim=True).squeeze(2)
        )
        return torch.cat(
            [deriv_wrt_qr, deriv_wrt_qi, deriv_wrt_qj, torch.zeros_like(deriv_wrt_qr)],
            dim=1,
        )

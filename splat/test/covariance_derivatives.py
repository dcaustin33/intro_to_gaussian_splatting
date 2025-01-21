"""
This is updated from the test_grads.ipynb as opposed to the covariance_derivatives.py 
file which is unchecked. I need to check all of these
"""

import torch

from splat.utils import build_rotation


def d_inv_wrt_a(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    grad_output: torch.Tensor,
):
    """
    All tensors are nx1 tensors - returns a nx1 tensor by summing over the last dimension
    """
    det = a * d - b * c
    deriv = -1 * (d**2) / (det**2) * grad_output[:, 0, 0]
    deriv += (b * d) / (det**2) * grad_output[:, 0, 1]
    deriv += (c * d) / (det**2) * grad_output[:, 1, 0]
    deriv += -1 * (b * c) / (det**2) * grad_output[:, 1, 1]
    return deriv


def d_inv_wrt_b(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    grad_output: torch.Tensor,
):
    det = a * d - b * c
    deriv = (c * d) / (det**2) * grad_output[:, 0, 0]
    deriv += -1 * (a * d) / (det**2) * grad_output[:, 0, 1]
    deriv += -1 * (c * c) / (det**2) * grad_output[:, 1, 0]
    deriv += (a * c) / (det**2) * grad_output[:, 1, 1]
    return deriv


def d_inv_wrt_c(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    grad_output: torch.Tensor,
):
    det = a * d - b * c
    deriv = (b * d) / (det**2) * grad_output[:, 0, 0]
    deriv += -1 * (b * b) / (det**2) * grad_output[:, 0, 1]
    deriv += -1 * (a * d) / (det**2) * grad_output[:, 1, 0]
    deriv += (a * b) / (det**2) * grad_output[:, 1, 1]
    return deriv


def d_inv_wrt_d(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    grad_output: torch.Tensor,
):
    det = a * d - b * c
    deriv = -1 * (b * c) / (det**2) * grad_output[:, 0, 0]
    deriv += (a * b) / (det**2) * grad_output[:, 0, 1]
    deriv += (a * c) / (det**2) * grad_output[:, 1, 0]
    deriv += -1 * (a * a) / (det**2) * grad_output[:, 1, 1]
    return deriv


class invert_2x2_matrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix: torch.Tensor):
        """input is a nx2x2 tensor, returns the inverse of each 2x2 matrix"""
        ctx.save_for_backward(matrix)
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

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """grad_output is a nx2x2 tensor, returns the gradient of the inverse of each 2x2 matrix"""
        matrix = ctx.saved_tensors[0]
        a = matrix[:, 0, 0]
        b = matrix[:, 0, 1]
        c = matrix[:, 1, 0]
        d = matrix[:, 1, 1]
        grad_a = d_inv_wrt_a(a, b, c, d, grad_output)
        grad_b = d_inv_wrt_b(a, b, c, d, grad_output)
        grad_c = d_inv_wrt_c(a, b, c, d, grad_output)
        grad_d = d_inv_wrt_d(a, b, c, d, grad_output)
        return torch.stack([grad_a, grad_b, grad_c, grad_d], dim=-1).view(matrix.shape)


class covariance_3d_to_covariance_2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, covariance_3d: torch.Tensor, U: torch.Tensor):
        """
        Covariance 3d is the nx3x3 covariance matrix.
        U is the J@W.T matrix. this is a 3x3 matrix
        To get the covariance 2d we do U.T @ covariance_3d @ U
        """
        ctx.save_for_backward(U, covariance_3d)
        outcome = torch.bmm(U.transpose(1, 2), torch.bmm(covariance_3d, U))
        return outcome

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        U, covariance_3d = ctx.saved_tensors

        # Derivative of (U^T * C * U) w.r.t. C = U * grad_output * U^T
        # grad_cov3d = torch.einsum("nij,njk->nik", U, grad_output)
        # grad_cov3d = torch.einsum("nij,njk->nik", grad_cov3d, U.transpose(1, 2))
        grad_cov3d = U @ grad_output @ U.transpose(1, 2)
        # Derivative of (U^T * C * U) w.r.t. U
        # Z = (U^T * (C * U)) Y= C * U
        # the contribution from Y is covariance_3d.T @ grad_output
        y = torch.einsum("nij,njk->nik", covariance_3d, U)
        deriv_U_first_part = torch.einsum("nij,njk->nik", grad_output, y.transpose(1, 2)).transpose(1, 2)
        dz_dy = torch.einsum("nij,njk->nik", U, grad_output)
        dy_du = torch.einsum("nij,njk->nik", covariance_3d.transpose(1, 2), dz_dy)
        return grad_cov3d, deriv_U_first_part + dy_du


class R_S_To_M(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R: torch.Tensor, S: torch.Tensor):
        """R is a nx3x3 rotation matrix, S is a nx3x3 scale matrix"""
        ctx.save_for_backward(R, S)
        return torch.einsum("nij,njk->nik", R, S)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        R, S = ctx.saved_tensors
        grad_R = torch.einsum("nij,njk->nik", grad_output, S.transpose(1, 2))
        grad_S = torch.einsum("nij,njk->nik", R.transpose(1, 2), grad_output)
        return grad_R, grad_S


class M_To_Covariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M: torch.Tensor):
        """Normal would be M@M.T but equivalent for our scenario"""
        ctx.save_for_backward(M)
        return M.pow(2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        M = ctx.saved_tensors[0]
        return 2 * grad_output * M


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


def d_r_wrt_qk(quats: torch.Tensor, n: int) -> torch.Tensor:
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
    derivative[:, 0, 0] = -2 * qk
    derivative[:, 0, 1] = -qr
    derivative[:, 0, 2] = qi
    derivative[:, 1, 0] = qr
    derivative[:, 1, 1] = -2 * qk
    derivative[:, 1, 2] = qj
    derivative[:, 2, 0] = qi
    derivative[:, 2, 1] = qj
    derivative[:, 2, 2] = 0
    return 2 * derivative


class quats_to_R(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quats: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(quats)
        R = build_rotation(quats, normalize=False)
        return R

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        grad output is nx3x3 and the jacobian nx3x3x4 so deriv_wrt_qr
        is nx3x3 where each entry is the derivative wrt one element in the final wrt r
        """
        quats = ctx.saved_tensors[0]
        deriv_wrt_qr = d_r_wrt_qr(quats, quats.shape[0])
        deriv_wrt_qi = d_r_wrt_qi(quats, quats.shape[0])
        deriv_wrt_qj = d_r_wrt_qj(quats, quats.shape[0])
        deriv_wrt_qk = d_r_wrt_qk(quats, quats.shape[0])

        deriv_wrt_qr = (
            (grad_output * deriv_wrt_qr).sum(dim=(1, 2), keepdim=True).squeeze(2)
        )
        deriv_wrt_qi = (
            (grad_output * deriv_wrt_qi).sum(dim=(1, 2), keepdim=True).squeeze(2)
        )
        deriv_wrt_qj = (
            (grad_output * deriv_wrt_qj).sum(dim=(1, 2), keepdim=True).squeeze(2)
        )
        deriv_wrt_qk = (
            (grad_output * deriv_wrt_qk).sum(dim=(1, 2), keepdim=True).squeeze(2)
        )
        return torch.cat(
            [deriv_wrt_qr, deriv_wrt_qi, deriv_wrt_qj, deriv_wrt_qk], dim=1
        )


class normalize_quats(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quats: torch.Tensor):
        """Quats are a nx4 tensor"""
        ctx.save_for_backward(quats)
        return quats / quats.norm(dim=1, keepdim=True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        quats = ctx.saved_tensors[0]
        norm = quats.norm(dim=1, keepdim=True)
        norm_cube = norm**3

        quats_outer = torch.einsum("ni,nj->nij", quats, quats)
        eye = torch.eye(4, dtype=quats.dtype, device=quats.device).unsqueeze(0)

        jacobian = (eye / norm.unsqueeze(2)) - (quats_outer / norm_cube.unsqueeze(2))
        grad_input = torch.einsum("nij,nj->ni", jacobian, grad_output)
        return grad_input


class scale_to_s_matrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s: torch.Tensor):
        """Takes the nx3 tensor and returns the nx3x3 diagonal matrix"""
        return torch.diag_embed(s)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Grad output is a nx3x3 tensor"""
        deriv_wr_s1 = grad_output[:, 0, 0].view(-1, 1)
        deriv_wr_s2 = grad_output[:, 1, 1].view(-1, 1)
        deriv_wr_s3 = grad_output[:, 2, 2].view(-1, 1)
        return torch.cat([deriv_wr_s1, deriv_wr_s2, deriv_wr_s3], dim=1)


def r_s_to_cov_2d(r: torch.Tensor, s: torch.Tensor, U: torch.Tensor):
     """
     r is a unnormalized nx4 tensor, s is a nx3 tensor
     U is the J@W matrix. this is a 3x3 matrix.
     In our example its W.t since we are using the OpenGL convention
     """
     r = normalize_quats.apply(r)
     R = quats_to_R.apply(r)
     S = scale_to_s_matrix.apply(s)
     M = R_S_To_M.apply(R, S)
     cov_3d = M_To_Covariance.apply(M)
     cov_2d = covariance_3d_to_covariance_2d.apply(cov_3d, U)
     inv_cov_2d = invert_2x2_matrix.apply(cov_2d[:, :2, :2])
     return inv_cov_2d
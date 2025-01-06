"""
The purpose of this file is to go from quaternions (normalized) and scales (after activation) to the 3D covariance matrix
"""
import torch
from torch.autograd.gradcheck import gradcheck

from splat.utils import build_rotation


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
    derivative[:, 1, 1] = -2*qk
    derivative[:, 1, 2] = qj
    derivative[:, 2, 0] = qi
    derivative[:, 2, 1] = qj
    derivative[:, 2, 2] = 0
    return 2 * derivative

class quatsToR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quats: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(quats)
        R = build_rotation(quats, normalize=False)
        return R
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        quats = ctx.saved_tensors[0]
        deriv_wrt_qr = d_r_wrt_qr(quats, quats.shape[0])
        deriv_wrt_qi = d_r_wrt_qi(quats, quats.shape[0])
        deriv_wrt_qj = d_r_wrt_qj(quats, quats.shape[0])
        deriv_wrt_qk = d_r_wrt_qk(quats, quats.shape[0])
        
        deriv_wrt_qr = (grad_output * deriv_wrt_qr).sum(dim=(1, 2), keepdim=True).squeeze(2)
        deriv_wrt_qi = (grad_output * deriv_wrt_qi).sum(dim=(1, 2), keepdim=True).squeeze(2)
        deriv_wrt_qj = (grad_output * deriv_wrt_qj).sum(dim=(1, 2), keepdim=True).squeeze(2)
        deriv_wrt_qk = (grad_output * deriv_wrt_qk).sum(dim=(1, 2), keepdim=True).squeeze(2)
        return torch.cat([deriv_wrt_qr, deriv_wrt_qi, deriv_wrt_qj, deriv_wrt_qk], dim=1)
    
class s_to_S(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s: torch.Tensor) -> torch.Tensor:
        """
        s is nx3 tensor
        S is nx3x3 tensor
        """
        ctx.save_for_backward(s)
        S = torch.diag_embed(s)
        return S
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        grad_output is nx3x3 tensor
        s is nx3 tensor
        """
        s = ctx.saved_tensors[0]
        deriv_wrt_s1 = grad_output[:, 0:1, 0:1].view(-1, 1)
        deriv_wrt_s2 = grad_output[:, 1:2, 1:2].view(-1, 1)
        deriv_wrt_s3 = grad_output[:, 2:3, 2:3].view(-1, 1)
        return torch.cat([deriv_wrt_s1, deriv_wrt_s2, deriv_wrt_s3], dim=1)
    
class R_S_to_M(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        R is nx3x3 tensor
        S is nx3x3 tensor
        M is nx3x3 tensor
        """
        ctx.save_for_backward(R, S)
        M = torch.bmm(R, S)
        return M
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        R, S = ctx.saved_tensors
        deriv_wrt_R = torch.bmm(grad_output, S.transpose(1, 2))
        deriv_wrt_S = torch.bmm(R.transpose(1, 2), grad_output)
        
        return deriv_wrt_R, deriv_wrt_S
    
class M_to_covariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(M)
        covariance = M.pow(2)
        return covariance
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """So the grad output should be nx3x3 and M is nx3x3"""
        # TODO I am unsure how to use this tensor - 
        # the paper says something different than what I am getting
        M = ctx.saved_tensors[0]
        deriv_wrt_M = 2 * grad_output * M
        return deriv_wrt_M


class quats_s_to_covariance2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quats: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        quats is nx4 tensor
        s is nx3 tensor
        
        Returns a nx3x3 covariance matrix
        """
        ctx.save_for_backward(quats, s)
        R = quatsToR.apply(quats)
        S = s_to_S.apply(s)
        M = R_S_to_M.apply(R, S)
        covariance = M_to_covariance.apply(M)
        return covariance
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        grad_output is nx3x3 tensor
        Returns a tuple of the derivatives of the covariance matrix wrt the 
        quats (n, 4) and s (n, 3)
        """
        quats, s = ctx.saved_tensors
        deriv_wrt_quats = quatsToR.backward(quats, grad_output)
        deriv_wrt_s = s_to_S.backward(s, grad_output)
        return deriv_wrt_quats, deriv_wrt_s
    
    

def quats_s_to_covariance(quats: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Function to compute the covariance matrix.
    It assumes that the quats are normalized and the s is post activation
    
    quats is nx4 tensor
    s is nx3 tensor
    
    Returns a nx3x3 covariance matrix
    """
    R = quatsToR.apply(quats)
    S = s_to_S.apply(s)
    M = R_S_to_M.apply(R, S)
    return M_to_covariance.apply(M)
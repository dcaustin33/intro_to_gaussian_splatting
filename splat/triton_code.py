import triton
import triton.language as tl
import torch

BLOCK_SIZE = 16

@triton.jit
def compute_gaussian_weight(
    point_means: torch.Tensor,
    points_in_view: torch.Tensor,
    inverse_covariance: torch.Tensor,
    
) -> None:
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
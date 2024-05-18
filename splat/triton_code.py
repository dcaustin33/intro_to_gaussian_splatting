import os
os.environ['TRITON_INTERPRET'] = '1'

import triton
import triton.language as tl
import torch

BLOCK_SIZE = 16

def check_tensors_gpu_ready(*tensors):
    for t in tensors:
        assert t.is_contiguous, "A tensor is not contiguous"
        if not os.environ.get('TRITON_INTERPRET') == '1': assert t.is_cuda, "A tensor is not on cuda"
        


@triton.jit
def compute_gaussian_weight(
    point_means,
    points_in_view,
    inverse_covariance,
    point_weight,
    num_points: tl.constexpr,
    zero: tl.constexpr
) -> None:
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    # each block is loading all the points in view
    points_in_view_offsets = tl.arange(0, num_points * 2)
    mask = points_in_view_offsets < num_points
    real_points_in_view = tl.load(points_in_view + points_in_view_offsets) > 0
    
    points_x = tl.load(point_means + points_in_view_offsets * 2, mask=real_points_in_view)
    points_y = tl.load(point_means + points_in_view_offsets * 2 + 1, mask=real_points_in_view)
    import pdb; pdb.set_trace()
    
    # pixel_coord = torch.Tensor([pid_0, pid_1]).to("cpu")
    # difference = point_means - pixel_coord
    # weight = tl.dot(tl.dot(difference, inverse_covariance), tl.transpose(difference))
    # weight = tl.exp(-0.5 * weight)
    # tl.store(point_weight + points_in_view_offsets, weight)

    
    
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_points = 10
    point_means = torch.rand_like(torch.zeros(num_points, 2, device=device))
    points_in_view = torch.randint(0, 2, (num_points, 1), device=device, dtype=torch.int32)
    inverse_covariance = torch.rand_like(torch.zeros(num_points, 2, 2, device=device))
    
    point_weight = torch.zeros(num_points, 1, device=device)
    
    compute_gaussian_weight[1,](point_means, points_in_view, inverse_covariance, point_weight, num_points, 0)
    import pdb; pdb.set_trace()
    
    
    
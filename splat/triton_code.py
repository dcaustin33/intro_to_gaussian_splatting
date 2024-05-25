import os

os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 16


def check_tensors_gpu_ready(*tensors):
    for t in tensors:
        assert t.is_contiguous, "A tensor is not contiguous"
        if not os.environ.get("TRITON_INTERPRET") == "1":
            assert t.is_cuda, "A tensor is not on cuda"


@triton.jit
def compute_gaussian_weight(
    point_means,
    points_in_view,
    inverse_covariance,
    point_weight,
    num_points: tl.constexpr,
) -> None:
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # each block is loading all the points in view
    points_in_view_offsets = tl.arange(0, num_points)
    real_points_in_view = tl.load(points_in_view + points_in_view_offsets) > 0

    inverse_covariance_x = tl.load(
        inverse_covariance + points_in_view_offsets * 4 + 0, mask=real_points_in_view
    )
    inverse_covariance_y = tl.load(
        inverse_covariance + points_in_view_offsets * 4 + 3, mask=real_points_in_view
    )
    inverse_covariance_xy = tl.load(
        inverse_covariance + points_in_view_offsets * 4 + 2, mask=real_points_in_view
    )
    points_x = tl.load(
        point_means + points_in_view_offsets * 2, mask=real_points_in_view
    )
    points_y = tl.load(
        point_means + points_in_view_offsets * 2 + 1, mask=real_points_in_view
    )

    difference_x = points_x - pid_0
    difference_y = points_y - pid_1
    difference_first = (
        difference_x * inverse_covariance_x + difference_y * inverse_covariance_xy
    )
    difference_second = (
        difference_x * inverse_covariance_xy + difference_y * inverse_covariance_y
    )
    final = difference_first * difference_x + difference_second * difference_y
    tl.store(point_weight + points_in_view_offsets, final, mask=real_points_in_view)


@triton.jit
def render_tile(
    points,
    points_in_view,
    inverse_covariance,
    point_weight,
    block_size: tl.constexpr,
) -> None:
    """
    Triton kernel that renders a full image tile.
    Takes in all the points, only loads those that are in view.
    Points are in sorted order so
    """


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    """Test function that adds two vectors in memory."""
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_points = 4
    point_means = torch.Tensor([[0, 0], [1, 1], [2, 2], [3, 3]]).to(device)
    points_in_view = torch.Tensor([1, 1, 1, 1]).to(device)
    # convert to torch.int32
    points_in_view = points_in_view.to(torch.int32)
    inverse_covariance = (
        torch.Tensor(
            [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]]
        )
        .to(device)
        .contiguous()
    )

    point_weight = torch.zeros(num_points, 1, device=device)

    compute_gaussian_weight[1,](
        point_means, points_in_view, inverse_covariance, point_weight, num_points
    )

    # n_elements = 10
    # x = torch.Tensor([0, 1, 2, 3])
    # y = torch.Tensor([4, 5, 6, 7])
    # output = torch.zeros_like(x)
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # # NOTE:
    # #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    # #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    # #  - Don't forget to pass meta-parameters as keywords arguments.
    # add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

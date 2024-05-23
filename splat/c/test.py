from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

def load_cpp_extension():
    cpp_source = Path(
        "/Users/derek/Desktop/personal_gaussian_splatting/splat/c/loop.c"
    ).read_text()

    # Load the CUDA kernel as a PyTorch extension
    copy_extension = load_inline(
        name="render_image",
        cpp_sources=cpp_source,
        functions=["render_pixel", "render_image"],
        with_cuda=False,
        # cuda_sources=cuda_source,
        # extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return copy_extension


def main():
    """
    Use torch cpp inline extension function to compile the kernel in grayscale_kernel.cu.
    Read input image, convert it to grayscale via custom cuda kernel and write it out as png.
    """

    x_means = torch.rand(10, 2)
    inverse_covariance = torch.eye(2).repeat(10, 1, 1) * 1
    opacity = torch.ones(10) * 0.5
    pixel_coords = torch.Tensor([0, 0])
    pixel_colors = torch.randint(0, 255, (10, 3))
    ext = load_cpp_extension()

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()

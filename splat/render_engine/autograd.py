import torch
import torch.autograd

if torch.cuda.is_available():
    from splat.c import render_tile_cuda, render_engine_backwards

class autograd_render_tile_cuda(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tile_size: int,
        means_3d: torch.Tensor,
        colors: torch.Tensor,
        unactivated_opacities: torch.Tensor,
        inverted_covariance_2d: torch.Tensor,
        image: torch.Tensor,
        starting_indices: torch.Tensor,
        final_tile_indices: torch.Tensor,
        array_indices: torch.Tensor,
        height: int,
        width: int,
        length_tiles_touched: int,
        length_array: int,
    ):
        """
        Inputs:
            tile_size: int
            means_3d: torch.Tensor, these are pixel coords with a z dimension
            colors: torch.Tensor, nx3 for rgb
            unactivated_opacities: torch.Tensor, nx1 - will need a sigmoid
            inverted_covariance_2d: torch.Tensor, nx2x2
            image: torch.Tensor, height x width x 3
            starting_indices: torch.Tensor, height * width
            final_tile_indices: torch.Tensor, length_array aka where render should stop
            array_indices: torch.Tensor, length_array aka mapping from 
                tile_indices to the actual gaussians
            height: int
            width: int
            length_tiles_touched: int
            length_array: int
        """
        ctx.save_for_backward(
            torch.tensor(tile_size),
            means_3d,
            colors,
            unactivated_opacities,
            inverted_covariance_2d,
            image,
            starting_indices,
            final_tile_indices,
            array_indices,
            torch.tensor(height),
            torch.tensor(width),
            torch.tensor(length_tiles_touched),
            torch.tensor(length_array),
        )

        image = render_tile_cuda.render_tile_cuda(
            tile_size,
            means_3d.contiguous(),
            colors.contiguous(),
            unactivated_opacities.contiguous(),
            inverted_covariance_2d.contiguous(),
            image.contiguous(),
            starting_indices.contiguous(),
            final_tile_indices.contiguous(),
            array_indices.contiguous(),
            height,
            width,
            length_tiles_touched,
            length_array,
        )
        return image
    
    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        """Grad output is the dl/dpixel"""

        tile_size = ctx.saved_tensors[0]
        means_3d = ctx.saved_tensors[1]
        colors = ctx.saved_tensors[2]
        unactivated_opacities = ctx.saved_tensors[3]
        inverted_covariance_2d = ctx.saved_tensors[4]
        image = ctx.saved_tensors[5]
        starting_indices = ctx.saved_tensors[6]
        final_tile_indices = ctx.saved_tensors[7]
        array_indices = ctx.saved_tensors[8]
        height = ctx.saved_tensors[9]
        width = ctx.saved_tensors[10]
        length_tiles_touched = ctx.saved_tensors[11]
        length_array = ctx.saved_tensors[12]

        gaussian_mean_3d_grad = torch.zeros_like(means_3d).to(means_3d.device)
        color_grad = torch.zeros_like(colors).to(colors.device)
        opacity_grad = torch.zeros_like(unactivated_opacities).to(unactivated_opacities.device)
        inverted_covariance_2d_grad = torch.zeros_like(inverted_covariance_2d).to(inverted_covariance_2d.device)
        
        render_engine_backwards.render_tile_cuda_backwards(
            int(tile_size.item()),
            means_3d.contiguous(),
            colors.contiguous(),
            unactivated_opacities.contiguous(),
            inverted_covariance_2d.contiguous(),
            image.contiguous(),
            starting_indices.contiguous(),
            final_tile_indices.contiguous(),
            array_indices.contiguous(),
            int(height.item()),
            int(width.item()),
            int(length_tiles_touched.item()),
            int(length_array.item()),
            grad_output.contiguous(),
            gaussian_mean_3d_grad.contiguous(),
            color_grad.contiguous(),
            opacity_grad.contiguous(),
            inverted_covariance_2d_grad.contiguous(),
        )

        return (
            None,
            gaussian_mean_3d_grad,
            color_grad,
            opacity_grad,
            inverted_covariance_2d_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        
        
        
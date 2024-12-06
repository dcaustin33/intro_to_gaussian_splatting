#include <torch/extension.h>

#include <cmath>  // Include this header for expf function
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

#include <pybind11/pybind11.h>

// #define PRINT_DEBUG
#define TILE_SIZE 16
#define CUDA_CHECK(call)                                                                   \
    {                                                                                      \
        cudaError_t err = call;                                                            \
        if (err != cudaSuccess)                                                            \
        {                                                                                  \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err));                             \
        }                                                                                  \
    }
#define CHECK_CUDA_INPUT(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS_INPUT(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)   \
    CHECK_CUDA_INPUT(x); \
    CHECK_CONTIGUOUS_INPUT(x)

namespace py = pybind11;

__global__ void get_start_idx_kernel(
    float* array,
    int* starting_idx,
    int total_x_tiles,
    int total_y_tiles,
    int array_length
)
{
    // we are only going to use 1d blocks and 1d thread blocks
    int array_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (array_idx >= array_length)
    {
        return;
    }

    int tile_x = static_cast<int>(array[4 * array_idx]);
    int tile_y = static_cast<int>(array[4 * array_idx + 1]);

    // Compute linear index into array_map
    int map_idx = tile_y * total_x_tiles + tile_x;
    int *ptr = &starting_idx[map_idx];

    // Try to set this position if it's -1
    int old_val = atomicCAS(ptr, -1, array_idx);

    if (old_val != -1) {
        // If the slot wasn't -1, we only update if idx < old_val
        // old_val here is what was previously at *ptr before CAS
        // We must re-check the current value in *ptr, since atomicCAS 
        // could have changed it if another thread updated in between.
        int cur_val = atomicAdd(ptr, 0); // atomicAdd with 0 to read atomically
        if (array_idx < cur_val) {
            // Use atomicMin to attempt to reduce the value
            atomicMin(ptr, array_idx);
        }
    }
}


torch::Tensor get_start_idx_cuda(
    torch::Tensor array,
    int total_x_tiles,
    int total_y_tiles
)
{
    CHECK_INPUT(array);
    torch::Tensor starting_idx = torch::zeros({total_y_tiles, total_x_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(array.device()));
    int array_length = array.size(0);
    dim3 grid_size((array_length + TILE_SIZE*TILE_SIZE - 1) / (TILE_SIZE*TILE_SIZE), 1);
    dim3 block_size(TILE_SIZE * TILE_SIZE, 1);
    get_start_idx_kernel<<<grid_size, block_size>>>(
        array.data_ptr<float>(),
        starting_idx.data_ptr<int>(),
        total_x_tiles,
        total_y_tiles,
        array_length
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return starting_idx;
}

PYBIND11_MODULE(preprocessing, m)
{
    m.def("get_start_idx_cuda", &get_start_idx_cuda, "Get the start idx of the tile");
}
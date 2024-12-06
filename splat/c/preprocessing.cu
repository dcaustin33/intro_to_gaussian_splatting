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

// #define DEBUG_PRINT

namespace py = pybind11;

__global__ void get_start_idx_kernel(
    at::Half* array,
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

    float val_x = static_cast<float>(array[4 * array_idx]);
    float val_y = static_cast<float>(array[4 * array_idx + 1]);
    int tile_x = static_cast<int>(val_x);
    int tile_y = static_cast<int>(val_y);

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
#ifdef DEBUG_PRINT
    int target_tile = 1;
    if (map_idx == target_tile)
    {
        printf("array_idx: %d val_x: %f val_y: %f tile_x: %d tile_y: %d old_val: %d, array_length: %d\n", 
            array_idx, val_x, val_y, tile_x, tile_y, old_val, array_length
        );

    }
#endif
}


torch::Tensor get_start_idx_cuda(
    torch::Tensor array,
    int total_x_tiles,
    int total_y_tiles
)
{
    CHECK_INPUT(array);
    torch::Tensor starting_idx = torch::ones({total_y_tiles, total_x_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(array.device())) * -1;
    int array_length = array.size(0);
    dim3 grid_size((array_length + TILE_SIZE*TILE_SIZE - 1) / (TILE_SIZE*TILE_SIZE), 1);
    dim3 block_size(TILE_SIZE * TILE_SIZE, 1);
    get_start_idx_kernel<<<grid_size, block_size>>>(
        array.data_ptr<at::Half>(),
        starting_idx.data_ptr<int>(),
        total_x_tiles,
        total_y_tiles,
        array_length
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return starting_idx;
}


__global__ void create_key_to_tile_map_kernel(
    at::Half* array,
    float* means_3d,
    int* prefix_sum,
    int* top_left,
    int* bottom_right,
    int prefix_sum_length
) 
{
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (start_idx >= prefix_sum_length) {
        return;
    }
    // prefix sum has a value in first entry not 0
    int array_idx;
    if (start_idx == 0) {
        array_idx = 0;
    } else {
        array_idx = prefix_sum[start_idx - 1];
    }

    // 2xn so you have to wrap all of the way around
    int top_left_x = static_cast<int>(top_left[start_idx]);
    int top_left_y = static_cast<int>(top_left[start_idx + prefix_sum_length]);
    int bottom_right_x = static_cast<int>(bottom_right[start_idx]);
    int bottom_right_y = static_cast<int>(bottom_right[start_idx + prefix_sum_length]);
#ifdef DEBUG_PRINT
    int target_array_idx = 0;
    if (start_idx == target_array_idx)
    {
        printf("start_idx: %d top_left_x: %d top_left_y: %d bottom_right_x: %d bottom_right_y: %d\n", 
            start_idx, top_left_x, top_left_y, bottom_right_x, bottom_right_y
        );
    }
#endif
    float z_depth = means_3d[start_idx * 4 + 2];

    for (int x = top_left_x; x <= bottom_right_x; x++) {
        for (int y = top_left_y; y <= bottom_right_y; y++) {
            array[4 * array_idx] = static_cast<at::Half>(x);
            array[4 * array_idx + 1] = static_cast<at::Half>(y);
            array[4 * array_idx + 2] = static_cast<at::Half>(z_depth);
            array[4 * array_idx + 3] = static_cast<at::Half>(start_idx);
            array_idx++;
        }
    }

}


torch::Tensor create_key_to_tile_map_cuda(
    torch::Tensor array,
    torch::Tensor means_3d,
    torch::Tensor top_left,
    torch::Tensor bottom_right,
    torch::Tensor prefix_sum
) 
{
    CHECK_INPUT(array);
    CHECK_INPUT(means_3d);
    CHECK_INPUT(top_left);
    CHECK_INPUT(bottom_right);
    CHECK_INPUT(prefix_sum);

    int prefix_sum_length = prefix_sum.size(0);
    dim3 grid_size((prefix_sum_length + TILE_SIZE*TILE_SIZE - 1) / (TILE_SIZE*TILE_SIZE), 1);
    dim3 block_size(TILE_SIZE * TILE_SIZE, 1);
    create_key_to_tile_map_kernel<<<grid_size, block_size>>>(
        array.data_ptr<at::Half>(),
        means_3d.data_ptr<float>(),
        prefix_sum.data_ptr<int>(),
        top_left.data_ptr<int>(),
        bottom_right.data_ptr<int>(),
        prefix_sum_length
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return array;
}

PYBIND11_MODULE(preprocessing, m)
{
    m.def("get_start_idx_cuda", &get_start_idx_cuda, "Get the start idx of the tile");
    m.def("create_key_to_tile_map_cuda", &create_key_to_tile_map_cuda, "Create the key to tile map");
}
#include <torch/extension.h>

#include <cmath>  // Include this header for expf function
#include <cstdio>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>

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

void check_inputs(
    torch::Tensor point_means,
    torch::Tensor point_colors,
    torch::Tensor point_opacities,
    torch::Tensor inverse_covariance_2d,
    torch::Tensor image,
    torch::Tensor starting_tile_indices,
    torch::Tensor tile_idx,
    torch::Tensor array_indices)
{
    CHECK_INPUT(point_means);
    CHECK_INPUT(point_colors);
    CHECK_INPUT(point_opacities);
    CHECK_INPUT(inverse_covariance_2d);
    CHECK_INPUT(image);
    CHECK_INPUT(starting_tile_indices);
    CHECK_INPUT(tile_idx);
    CHECK_INPUT(array_indices);
}

namespace py = pybind11;

__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

__device__ float compute_pixel_strength(
    int pixel_x,
    int pixel_y,
    int point_x,
    int point_y,
    float inverse_covariance_a,
    float inverse_covariance_b,
    float inverse_covariance_c)
{
    // Compute the distance between the pixel and the point
    float dx = pixel_x - point_x;
    float dy = pixel_y - point_y;
    float power = dx * inverse_covariance_a * dx +
                  2 * dx * dy * inverse_covariance_b +
                  dy * dy * inverse_covariance_c;
    if (power < 0)
    {
        // according to chatgpt indicates numerical 
        // instability as this should never occur
        return 0.0f;
    }
    return exp(-0.5f * power);
}

__global__ void render_tile_kernel(
    int tile_size,
    float* point_means,
    float* point_colors,
    float* point_opacities,
    float* inverse_covariance_2d,
    float* image,
    int* starting_tile_indices,
    int* tile_idx,
    int* array_indices,
    int image_height,
    int image_width,
    int num_points,
    int num_array_points)
{
    if (tile_size != TILE_SIZE)
    {
        printf("Error: Tile size must be %d but got %d\n", TILE_SIZE, tile_size);
        return;
    }
    // so we need to load all the points
    // then each will have shared memory corresponding to
    // means, color, opacity, covariance, and then the tile id
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int pixel_x = threadIdx.x + blockIdx.x * tile_size;
    int pixel_y = threadIdx.y + blockIdx.y * tile_size;
    bool done = false;

    __shared__ float shared_point_means[TILE_SIZE * TILE_SIZE * 2];
    __shared__ bool shared_done_indicator[TILE_SIZE * TILE_SIZE];
    __shared__ float shared_point_colors[TILE_SIZE * TILE_SIZE * 3];
    __shared__ float shared_point_opacities[TILE_SIZE * TILE_SIZE];
    __shared__ float shared_inverse_covariance_2d[TILE_SIZE * TILE_SIZE * 3];

    if (pixel_x >= image_width || pixel_y >= image_height)
    {
        // still helps with the shared memory so no return
        done = true;
    }

    int target_pixel_x = 0;
    int target_pixel_y = 0;
    int target_tile_x = target_pixel_x / TILE_SIZE;
    int target_tile_y = target_pixel_y / TILE_SIZE;

    // then we have to load and if their tile does not match we indicate done in
    // the array
    int thread_dim = blockDim.x * blockDim.y;
    int num_y_tiles = gridDim.y;
    int round_counter = 0;
    int point_idx;
    float total_weight = 1.0f;
    float3 color = {0.0f, 0.0f, 0.0f};
    int num_done = 0;
    int correct_tile_idx = tile_x * num_y_tiles + tile_y;
    int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
    // print the max thread idx
    shared_done_indicator[thread_idx] = false;

    if (starting_tile_indices[correct_tile_idx] == -1)
    {
        return;
    }

    while (true)
    {
        __syncthreads();
        num_done = __syncthreads_count(done);
        if (num_done == thread_dim)
            break;


        // Calculate global point index for this round
        point_idx = starting_tile_indices[correct_tile_idx] + round_counter * thread_dim;
        if (point_idx < 0)
        {
            shared_done_indicator[thread_idx] = true;
        }

        // Calculate global memory offset for this point
        int point_offset = point_idx + thread_idx;
        if (point_offset >= num_array_points)
        {
            shared_done_indicator[thread_idx] = true;
        }
        else
        {
            int processed_gaussians_idx = array_indices[point_offset];
            if (processed_gaussians_idx >= num_points || processed_gaussians_idx < 0)
            {
                shared_done_indicator[thread_idx] = true;
            }
            else
            {
                // Load point data into shared memory
                shared_point_means[thread_idx * 2] = point_means[processed_gaussians_idx * 2];
                shared_point_means[thread_idx * 2 + 1] = point_means[processed_gaussians_idx * 2 + 1];
                shared_point_colors[thread_idx * 3] = point_colors[processed_gaussians_idx * 3];
                shared_point_colors[thread_idx * 3 + 1] = point_colors[processed_gaussians_idx * 3 + 1];
                shared_point_colors[thread_idx * 3 + 2] = point_colors[processed_gaussians_idx * 3 + 2];
                shared_point_opacities[thread_idx] = point_opacities[processed_gaussians_idx];

                // no need to store the other covariance as its symmetric
                shared_inverse_covariance_2d[thread_idx * 3] = inverse_covariance_2d[processed_gaussians_idx * 4];
                shared_inverse_covariance_2d[thread_idx * 3 + 1] = inverse_covariance_2d[processed_gaussians_idx * 4 + 1];
                shared_inverse_covariance_2d[thread_idx * 3 + 2] = inverse_covariance_2d[processed_gaussians_idx * 4 + 3];
            }

            if (tile_idx[point_offset] != correct_tile_idx)
            {
                shared_done_indicator[thread_idx] = true;
            }
        }

        // wait for all the memory loads to finish
        __syncthreads();
        round_counter++;

        if (pixel_x == target_pixel_x && pixel_y == target_pixel_y)
        {
            printf("round_counter: %d, done: %d\n", round_counter, done);
        }

        int shared_done_count = 0;
        if (!done)
        {
            // render the pixel by iterating through all points until weight or
            // a done indicator is reached
            for (int i = 0; i < thread_dim; i++)
            {
                if (shared_done_indicator[i])
                {
                    shared_done_count++;
                    continue;
                }
                else
                {
                    float weight = compute_pixel_strength(
                        pixel_x,
                        pixel_y,
                        shared_point_means[i * 2],
                        shared_point_means[i * 2 + 1],
                        shared_inverse_covariance_2d[i * 3],
                        shared_inverse_covariance_2d[i * 3 + 1],
                        shared_inverse_covariance_2d[i * 3 + 2]);

                    float opacity_output = sigmoid(shared_point_opacities[i]);
                    float alpha_value = min(0.99f, weight * (1 - opacity_output));
                    float test_T = total_weight * alpha_value;
                    if (pixel_x == 0 && pixel_y == 0)
                    {
                        printf("test_T: %f, weight: %f, opacity: %f\n", test_T, weight, opacity);
                    }
                    if (test_T < 0.001f)
                    {
                        done = true;
                        continue;
                    }
                    color.x += total_weight * alpha_value * shared_point_colors[i * 3];
                    color.y += total_weight * alpha_value * shared_point_colors[i * 3 + 1];
                    color.z += total_weight * alpha_value * shared_point_colors[i * 3 + 2];
                    total_weight = test_T;
                }
            }
            if (shared_done_count == thread_dim)
            {
                // this will eventually cause breaking
                done = true;
            }
        }
    }

    if (pixel_x < image_width && pixel_y < image_height)
    {
        int pixel_idx = (pixel_x * image_width + pixel_y) * 3;
        if (pixel_idx + 2 < image_width * image_height * 3)
        {
            image[pixel_idx] = color.x;
            image[pixel_idx + 1] = color.y;
            image[pixel_idx + 2] = color.z;
        }
    }
}

torch::Tensor render_tile_cuda(
    int tile_size,
    torch::Tensor point_means,
    torch::Tensor point_colors,
    torch::Tensor point_opacities,
    torch::Tensor inverse_covariance_2d,
    torch::Tensor image,
    torch::Tensor starting_tile_indices,
    torch::Tensor tile_idx,
    torch::Tensor array_indices,
    int image_height,
    int image_width,
    int num_points,
    int num_array_points)
{
    check_inputs(
        point_means,
        point_colors,
        point_opacities,
        inverse_covariance_2d,
        image,
        starting_tile_indices,
        tile_idx,
        array_indices);
    if (tile_size != TILE_SIZE)
    {
        throw std::runtime_error("Tile size must be 16 or TILE_SIZE in c code must change");
    }
    dim3 block_size(tile_size, tile_size);
    int grid_size_x = (image_width + tile_size - 1) / tile_size;
    int grid_size_y = (image_height + tile_size - 1) / tile_size;
    dim3 grid_size(grid_size_x, grid_size_y);

    torch::Tensor image_new = torch::zeros({image_height, image_width, 3}, image.options());

    // print the amount of elements in image tensor
    printf("Number of elements in image tensor: %zu\n", image.numel());
    render_tile_kernel<<<grid_size, block_size>>>(
        tile_size,
        point_means.data_ptr<float>(),
        point_colors.data_ptr<float>(),
        point_opacities.data_ptr<float>(),
        inverse_covariance_2d.data_ptr<float>(),
        image.data_ptr<float>(),
        starting_tile_indices.data_ptr<int>(),
        tile_idx.data_ptr<int>(),
        array_indices.data_ptr<int>(),
        image_height,
        image_width,
        num_points,
        num_array_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return image;
}

PYBIND11_MODULE(render_tile_cuda, m)
{
    m.def("render_tile_cuda", &render_tile_cuda, "Render a tile of the image");
}
#include <cstdio>
#include <cmath> // Include this header for expf function
#include <torch/extension.h>

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
    float power = dx * inverse_covariance_a * dx + 2 * dx * dy * inverse_covariance_b + dy * dy * inverse_covariance_c;
    return expf(-0.5f * power);
}

__global__ void render_tile(
    int tile_size,
    int tile_start_idx float *point_means,
    float *point_colors,
    float *point_opacities,
    float *inverse_covariance_2d,
    float *image,
    int *tile_idx int image_height,
    int image_width)
{
    // so we need to load all the points
    // then each will have shared memory corresponding to
    // means, color, opacity, covariance, and then the tile id
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int pixel_x = threadIdx.x + tile_x * tile_size;
    int pixel_y = threadIdx.y + tile_y * tile_size;
    bool done = false;

    __shared__ float point_means[tile_size * tile_size * 2];
    __shared__ bool done_indicator[tile_size * tile_size];
    __shared__ float point_colors[tile_size * tile_size * 3];
    __shared__ float point_opacities[tile_size * tile_size];
    __shared__ float inverse_covariance_2d[tile_size * tile_size * 3];

    if (pixel_x >= image_width || pixel_y >= image_height)
    {
        // still helps with the shared memory
        done = true;
    }

    // then we have to load and if their tile does not match we indicate done in the array

    int thread_dim = blockDim.x * blockDim.y;
    int round_counter = 0;
    int point_idx;
    float total_weight = 1.0f;
    float3 color = {0.0f, 0.0f, 0.0f};
    int num_done = 0;

    while (true)
    {
        num_done = __syncthreads_count(done);
        if (num_done == thread_dim)
            break;
        point_idx = tile_start_idx + round_counter * thread_dim;
        point_means[threadIdx.x + threadIdx.y * blockDim.x] = point_means[point_idx * 2 + threadIdx.x + threadIdx.y * blockDim.x];
        point_colors[threadIdx.x + threadIdx.y * blockDim.x] = point_colors[point_idx * 3 + threadIdx.x + threadIdx.y * blockDim.x];
        point_opacities[threadIdx.x + threadIdx.y * blockDim.x] = point_opacities[point_idx + threadIdx.x + threadIdx.y * blockDim.x];
        inverse_covariance_2d[threadIdx.x + threadIdx.y * blockDim.x] = inverse_covariance_2d[point_idx * 3 + threadIdx.x + threadIdx.y * blockDim.x];
        if (tile_idx[point_idx + threadIdx.x + threadIdx.y * blockDim.x] != correctTileIDX) // TODO CALCULATE THIS
        {
            done_indicator[threadIdx.x + threadIdx.y * blockDim.x] = true;
        }

        // wait for all the memory loads to finish
        sync_threads();
        round_counter++;

        if (!done)
        {
            // render the pixel by iterating through all points until weight or a done indicator is reached
            for (int i = 0; i < thread_dim; i++)
            {
                if (done_indicator[i])
                {
                    done = true;
                    break;
                }
                float weight = compute_pixel_strength(pixel_x, pixel_y, point_means[i * 2], point_means[i * 2 + 1], inverse_covariance_2d[i * 3], inverse_covariance_2d[i * 3 + 1], inverse_covariance_2d[i * 3 + 2]);
                float test_T = total_weight * (1 - torch.sigmoid(point_opacities[i]) * weight);
                if (test_T < MINWEIGHT_FIX_THIS)
                {
                    done = true;
                    break;
                }
                color.x += test_T * point_colors[i * 3];
                color.y += test_T * point_colors[i * 3 + 1];
                color.z += test_T * point_colors[i * 3 + 2];
                total_weight = test_T;
            }
        }
    }
    image[pixel_y * image_width + pixel_x] = color;
}
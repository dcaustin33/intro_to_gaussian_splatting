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

__device__ void three_by_three_matrix_multiply(const float A[3][3],
                                               const float B[3][3],
                                               float C[3][3])
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            float sum = 0;
            for (int k = 0; k < 3; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

__device__ void three_by_three_matrix_transpose(const float A[3][3],
                                               float B[3][3])
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            B[i][j] = A[j][i];
        }
    }
}

__device__ inline float clampf(float x, float x_min, float x_max)
{
    return fminf(x_max, fmaxf(x_min, x));
}


__global__ void get_start_idx_kernel(
    float* array,
    int* starting_idx,
    int total_x_tiles,
    int total_y_tiles,
    int array_length)
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
    int* ptr = &starting_idx[map_idx];

    // Try to set this position if it's -1
    int old_val = atomicCAS(ptr, -1, array_idx);

    if (old_val != -1)
    {
        // If the slot wasn't -1, we only update if idx < old_val
        // old_val here is what was previously at *ptr before CAS
        // We must re-check the current value in *ptr, since atomicCAS
        // could have changed it if another thread updated in between.
        int cur_val = atomicAdd(ptr, 0);  // atomicAdd with 0 to read atomically
        if (array_idx < cur_val)
        {
            // Use atomicMin to attempt to reduce the value
            atomicMin(ptr, array_idx);
        }
    }
#ifdef DEBUG_PRINT
    int target_tile = 1;
    if (map_idx == target_tile)
    {
        printf("array_idx: %d val_x: %f val_y: %f tile_x: %d tile_y: %d old_val: %d, array_length: %d\n",
               array_idx, val_x, val_y, tile_x, tile_y, old_val, array_length);
    }
#endif
}

torch::Tensor get_start_idx_cuda(
    torch::Tensor array,
    int total_x_tiles,
    int total_y_tiles)
{
    CHECK_INPUT(array);
    torch::Tensor starting_idx = torch::ones({total_y_tiles, total_x_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(array.device())) * -1;
    int array_length = array.size(0);
    dim3 grid_size((array_length + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE), 1);
    dim3 block_size(TILE_SIZE * TILE_SIZE, 1);
    get_start_idx_kernel<<<grid_size, block_size>>>(
        array.data_ptr<float>(),
        starting_idx.data_ptr<int>(),
        total_x_tiles,
        total_y_tiles,
        array_length);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return starting_idx;
}

__global__ void create_key_to_tile_map_kernel(
    float* array,
    float* means_3d,
    int64_t* prefix_sum,
    int* top_left,
    int* bottom_right,
    int prefix_sum_length)
{
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (start_idx >= prefix_sum_length)
    {
        return;
    }
    // prefix sum has a value in first entry not 0
    int array_idx;
    if (start_idx == 0)
    {
        array_idx = 0;
    } else
    {
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
               start_idx, top_left_x, top_left_y, bottom_right_x, bottom_right_y);
    }
#endif
    float z_depth = means_3d[start_idx * 3 + 2];

    for (int x = top_left_x; x <= bottom_right_x; x++)
    {
        for (int y = top_left_y; y <= bottom_right_y; y++)
        {
            array[4 * array_idx] = static_cast<float>(x);
            array[4 * array_idx + 1] = static_cast<float>(y);
            array[4 * array_idx + 2] = static_cast<float>(z_depth);
            array[4 * array_idx + 3] = static_cast<float>(start_idx);
            array_idx++;
        }
    }
}

torch::Tensor create_key_to_tile_map_cuda(
    torch::Tensor array,
    torch::Tensor means_3d,
    torch::Tensor top_left,
    torch::Tensor bottom_right,
    torch::Tensor prefix_sum)
{
    CHECK_INPUT(array);
    CHECK_INPUT(means_3d);
    CHECK_INPUT(top_left);
    CHECK_INPUT(bottom_right);
    CHECK_INPUT(prefix_sum);

    int prefix_sum_length = prefix_sum.size(0);
    dim3 grid_size((prefix_sum_length + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE), 1);
    dim3 block_size(TILE_SIZE * TILE_SIZE, 1);
    create_key_to_tile_map_kernel<<<grid_size, block_size>>>(
        array.data_ptr<float>(),
        means_3d.data_ptr<float>(),
        prefix_sum.data_ptr<int64_t>(),
        top_left.data_ptr<int>(),
        bottom_right.data_ptr<int>(),
        prefix_sum_length);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return array;
}

__global__ void get_3d_covariance_matrix_kernel(
    float* quaternions,
    float* scales,
    float* covariance,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    float r = quaternions[4 * idx];
    float x = quaternions[4 * idx + 1];
    float y = quaternions[4 * idx + 2];
    float z = quaternions[4 * idx + 3];
    float sx = expf(scales[idx * 3]);
    float sy = expf(scales[idx * 3 + 1]);
    float sz = expf(scales[idx * 3 + 2]);

    // normalize the quaternion
    float norm = sqrtf(r * r + x * x + y * y + z * z);
    r /= norm;
    x /= norm;
    y /= norm;
    z /= norm;

    float rotation_matrix[3][3];
    rotation_matrix[0][0] = 1 - 2 * (y * y + z * z);
    rotation_matrix[0][1] = 2 * (x * y - r * z);
    rotation_matrix[0][2] = 2 * (x * z + r * y);
    rotation_matrix[1][0] = 2 * (x * y + r * z);
    rotation_matrix[1][1] = 1 - 2 * (x * x + z * z);
    rotation_matrix[1][2] = 2 * (y * z - r * x);
    rotation_matrix[2][0] = 2 * (x * z - r * y);
    rotation_matrix[2][1] = 2 * (y * z + r * x);
    rotation_matrix[2][2] = 1 - 2 * (x * x + y * y);

    float scale_matrix[3][3] = {
        {sx, 0, 0},
        {0, sy, 0},
        {0, 0, sz}};

    // manual matrix multiplication
    // we know all off diagonal elements are 0
    // can optimize this later
    float output1[3][3] = {0};
    three_by_three_matrix_multiply(rotation_matrix, scale_matrix, output1);
    float output2[3][3] = {0};
    three_by_three_matrix_multiply(output1, scale_matrix, output2);
    float transpose_rotation_matrix[3][3] = {0};
    float covariance_3d[3][3] = {0};
    three_by_three_matrix_multiply(
        output2,
        three_by_three_matrix_transpose(rotation_matrix, transpose_rotation_matrix),
        covariance_3d);

    covariance[idx * 9] = covariance_3d[0][0];
    covariance[idx * 9 + 1] = covariance_3d[0][1];
    covariance[idx * 9 + 2] = covariance_3d[0][2];
    covariance[idx * 9 + 3] = covariance_3d[1][0];
    covariance[idx * 9 + 4] = covariance_3d[1][1];
    covariance[idx * 9 + 5] = covariance_3d[1][2];
    covariance[idx * 9 + 6] = covariance_3d[2][0];
    covariance[idx * 9 + 7] = covariance_3d[2][1];
    covariance[idx * 9 + 8] = covariance_3d[2][2];
}

torch::Tensor get_3d_covariance_matrix_cuda(
    torch::Tensor quaternions,
    torch::Tensor scales)
{
    CHECK_INPUT(quaternions);
    CHECK_INPUT(scales);

    int n = quaternions.size(0);
    torch::Tensor covariance = torch::zeros({n, 9}, torch::TensorOptions().dtype(torch::kFloat32).device(quaternions.device()));
    dim3 grid_size((n + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE), 1);
    dim3 block_size(TILE_SIZE * TILE_SIZE, 1);
    get_3d_covariance_matrix_kernel<<<grid_size, block_size>>>(
        quaternions.data_ptr<float>(),
        scales.data_ptr<float>(),
        covariance.data_ptr<float>(),
        n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return covariance;
}

__global__ void get_2d_covariance_matrix_kernel(
    float* points_homogeneous,
    float* covariance_3d,
    float* extrinsic_matrix,
    float tan_fovX,
    float tan_fovY,
    float focal_x,
    float focal_y,
    float* covariance_2d,
    float* points_camera_space,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    float x = points_homogeneous[idx * 3];
    float y = points_homogeneous[idx * 3 + 1];
    float z = points_homogeneous[idx * 3 + 2];
    float4 point_camera_space;
    point_camera_space.x = x * extrinsic_matrix[0] + y * extrinsic_matrix[4] + z * extrinsic_matrix[8] + extrinsic_matrix[12];
    point_camera_space.y = x * extrinsic_matrix[1] + y * extrinsic_matrix[5] + z * extrinsic_matrix[9] + extrinsic_matrix[13];
    point_camera_space.z = x * extrinsic_matrix[2] + y * extrinsic_matrix[6] + z * extrinsic_matrix[10] + extrinsic_matrix[14];
    point_camera_space.w = extrinsic_matrix[3] + extrinsic_matrix[7] + extrinsic_matrix[11] + extrinsic_matrix[15];

    // saving the computation for later
    points_camera_space[idx * 4] = point_camera_space.x;
    points_camera_space[idx * 4 + 1] = point_camera_space.y;
    points_camera_space[idx * 4 + 2] = point_camera_space.z;
    points_camera_space[idx * 4 + 3] = point_camera_space.w;

    point_camera_space.x = point_camera_space.x / point_camera_space.z;
    point_camera_space.y = point_camera_space.y / point_camera_space.z;

    float x_clamped = clampf(point_camera_space.x, -1.3 * tan_fovX, 1.3 * tan_fovX) * point_camera_space.z;
    float y_clamped = clampf(point_camera_space.y, -1.3 * tan_fovY, 1.3 * tan_fovY) * point_camera_space.z;

    float j[3][3] = {
        {focal_x / z, 0, -(focal_x * x_clamped) / (z * z)},
        {0, focal_y / z, -(focal_y * y_clamped) / (z * z)},
        {0, 0, 0}};

    float w[3][3] = {
        {extrinsic_matrix[0], extrinsic_matrix[1], extrinsic_matrix[2]},
        {extrinsic_matrix[3], extrinsic_matrix[4], extrinsic_matrix[5]},
        {extrinsic_matrix[6], extrinsic_matrix[7], extrinsic_matrix[8]}};
    w[2][0] = extrinsic_matrix[6];
    w[2][1] = extrinsic_matrix[7];
    w[2][2] = extrinsic_matrix[8];

    float t[3][3] = {0};
    float j_transpose[3][3] = {0};
    three_by_three_matrix_multiply(w, three_by_three_matrix_transpose(j, j_transpose), t);

    float covariance_output1[3][3] = {0};
    float t_transpose[3][3] = {0};
    float kernel_covariance[3][3] = {
        {covariance_3d[idx * 9], covariance_3d[idx * 9 + 1], covariance_3d[idx * 9 + 2]},
        {covariance_3d[idx * 9 + 3], covariance_3d[idx * 9 + 4], covariance_3d[idx * 9 + 5]},
        {covariance_3d[idx * 9 + 6], covariance_3d[idx * 9 + 7], covariance_3d[idx * 9 + 8]}};
    three_by_three_matrix_multiply(three_by_three_matrix_transpose(t, t_transpose), kernel_covariance, covariance_output1);
    float covariance_output2[3][3] = {0};
    three_by_three_matrix_multiply(covariance_output1, t, covariance_output2);

    covariance_2d[idx * 4] = covariance_output2[0][0] + 0.3;
    covariance_2d[idx * 4 + 1] = covariance_output2[0][1];
    covariance_2d[idx * 4 + 2] = covariance_output2[1][0];
    covariance_2d[idx * 4 + 3] = covariance_output2[1][1] + 0.3;
}

std::tuple<torch::Tensor, torch::Tensor> get_2d_covariance_matrix_cuda(
    torch::Tensor points_homogeneous,
    torch::Tensor covariance_3d,
    torch::Tensor extrinsic_matrix,
    float tan_fovX,
    float tan_fovY,
    float focal_x,
    float focal_y)
{
    CHECK_INPUT(points_homogeneous);
    CHECK_INPUT(covariance_3d);
    CHECK_INPUT(extrinsic_matrix);

    int n = points_homogeneous.size(0);
    torch::Tensor covariance_2d = torch::zeros({n, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(points_homogeneous.device()));
    torch::Tensor points_camera_space = torch::zeros({n, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(points_homogeneous.device()));
    dim3 grid_size((n + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE), 1);
    dim3 block_size(TILE_SIZE * TILE_SIZE, 1);
    get_2d_covariance_matrix_kernel<<<grid_size, block_size>>>(
        points_homogeneous.data_ptr<float>(),
        covariance_3d.data_ptr<float>(),
        extrinsic_matrix.data_ptr<float>(),
        tan_fovX,
        tan_fovY,
        focal_x,
        focal_y,
        covariance_2d.data_ptr<float>(),
        points_camera_space.data_ptr<float>(),
        n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return std::make_tuple(covariance_2d, points_camera_space);
}

PYBIND11_MODULE(preprocessing, m)
{
    m.def("get_start_idx_cuda", &get_start_idx_cuda, "Get the start idx of the tile");
    m.def("create_key_to_tile_map_cuda", &create_key_to_tile_map_cuda, "Create the key to tile map");
    m.def("get_3d_covariance_matrix_cuda", &get_3d_covariance_matrix_cuda, "Get the 3d covariance matrix");
    m.def("get_2d_covariance_matrix_cuda", &get_2d_covariance_matrix_cuda, "Get the 2d covariance matrix");
}
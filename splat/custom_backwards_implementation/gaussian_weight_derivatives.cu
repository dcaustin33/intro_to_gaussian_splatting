// this will mirror the gaussian_weight_derivatives.py file
// it will consist mainly of individual __global__ functions that 
// will be stitched together in a broader backwards pass
// this will make it easier to debug and test
// goal is to test the same way I tested the custom pytorch implementation

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math_functions.h>
#include <torch/extension.h>

#define CUDA_CHECK(call)                                                                   \
    {                                                                                      \
        cudaError_t err = call;                                                            \
        if (err != cudaSuccess)                                                            \
        {                                                                                  \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err));                             \
        }                                                                                  \
    }


namespace py = pybind11;


__global__ void backward_final_color(
    const float* __restrict__ grad_output,
    const float* __restrict__ color,
    const float* __restrict__ current_T,
    const float* __restrict__ alpha,
    float* grad_color,
    float* grad_alpha,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Now idx is a valid index in [0, N-1].
    grad_color[idx * 3 + 0] = grad_output[idx * 3 + 0] * current_T[idx] * alpha[idx];
    grad_color[idx * 3 + 1] = grad_output[idx * 3 + 1] * current_T[idx] * alpha[idx];
    grad_color[idx * 3 + 2] = grad_output[idx * 3 + 2] * current_T[idx] * alpha[idx];

    grad_alpha[idx] = grad_output[idx * 3 + 0] * color[idx * 3 + 0] * current_T[idx]
                    + grad_output[idx * 3 + 1] * color[idx * 3 + 1] * current_T[idx]
                    + grad_output[idx * 3 + 2] * color[idx * 3 + 2] * current_T[idx];
}

void backward_final_color_launcher(
    torch::Tensor grad_output,
    torch::Tensor color,
    torch::Tensor current_T,
    torch::Tensor alpha,
    torch::Tensor grad_color,
    torch::Tensor grad_alpha
) {
    int N = color.size(0);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    backward_final_color<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        color.data_ptr<float>(),
        current_T.data_ptr<float>(),
        alpha.data_ptr<float>(),
        grad_color.data_ptr<float>(),
        grad_alpha.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();  // Ensure the kernel execution is completed
    CUDA_CHECK(cudaGetLastError());
}

__global__ void get_alpha_backward_device(
    const float* __restrict__ grad_output,
    const float* __restrict__ gaussian_strength,
    const float* __restrict__ unactivated_opacity,
    float* grad_gaussian_strength,
    float* grad_unactivated_opacity,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute the derivative of the sigmoid function
    float sigmoid = 1.0f / (1.0f + expf(-unactivated_opacity[idx]));
    float derivative_sigmoid = sigmoid * (1.0f - sigmoid);

    // Compute gradients
    grad_gaussian_strength[idx] = grad_output[idx] * sigmoid;
    grad_unactivated_opacity[idx] = grad_output[idx] * gaussian_strength[idx] * derivative_sigmoid;
}

void get_alpha_backward_launcher(
    torch::Tensor grad_output,
    torch::Tensor gaussian_strength,
    torch::Tensor unactivated_opacity,
    torch::Tensor grad_gaussian_strength,
    torch::Tensor grad_unactivated_opacity
) {
    int N = unactivated_opacity.size(0);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    get_alpha_backward_device<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        gaussian_strength.data_ptr<float>(),
        unactivated_opacity.data_ptr<float>(),
        grad_gaussian_strength.data_ptr<float>(),
        grad_unactivated_opacity.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();  // Ensure the kernel execution is completed
    CUDA_CHECK(cudaGetLastError());
}

__global__ void gaussian_exp_backward_device(
    const float* __restrict__ grad_output,
    const float* __restrict__ gaussian_weight,
    float* grad_gaussian_weight,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute the gradient with respect to gaussian_weight
    grad_gaussian_weight[idx] = grad_output[idx] * expf(gaussian_weight[idx]);
}

void gaussian_exp_backward_launcher(
    torch::Tensor grad_output,
    torch::Tensor gaussian_weight,
    torch::Tensor grad_gaussian_weight
) {
    int N = gaussian_weight.size(0);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    gaussian_exp_backward_device<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        gaussian_weight.data_ptr<float>(),
        grad_gaussian_weight.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();  // Ensure the kernel execution is completed
    CUDA_CHECK(cudaGetLastError());
}

__global__ void gaussian_weight_grad_inv_cov(
    const float* __restrict__ grad_output,
    const float* __restrict__ diff,
    float* grad_inv_cov,
    int n_dims,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute gradient w.r.t. inverted covariance: -0.5 * (diff.T @ diff) scaled by grad_output
    for (int i = 0; i < n_dims; ++i) {
        for (int j = 0; j < n_dims; ++j) {
            grad_inv_cov[idx * n_dims * n_dims + i * n_dims + j] = -0.5f * grad_output[idx] * diff[idx * n_dims + i] * diff[idx * n_dims + j];
        }
    }
}

void gaussian_weight_grad_inv_cov_launcher(
    torch::Tensor grad_output,
    torch::Tensor diff,
    torch::Tensor grad_inv_cov
) {
    int N = diff.size(0);
    int n_dims = diff.size(2);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    gaussian_weight_grad_inv_cov<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        diff.data_ptr<float>(),
        grad_inv_cov.data_ptr<float>(),
        n_dims,
        N
    );

    cudaDeviceSynchronize();  // Ensure the kernel execution is completed
    CUDA_CHECK(cudaGetLastError());
}

__global__ void gaussian_weight_grad_gaussian_mean(
    const float* __restrict__ grad_output,
    const float* __restrict__ diff,
    const float* __restrict__ inverted_covariance,
    float* grad_gaussian_mean,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float deriv_output_wrt_diff1[2] = {0.0f, 0.0f};
    float deriv_output_wrt_diff2[2] = {0.0f, 0.0f};

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            deriv_output_wrt_diff1[i] += inverted_covariance[i * 2 + j] * diff[idx * 2 + j];
            deriv_output_wrt_diff2[i] += inverted_covariance[j * 2 + i] * diff[idx * 2 + j];
        }
    }

    for (int i = 0; i < 2; ++i) {
        grad_gaussian_mean[idx * 2 + i] = 0.5 * grad_output[idx] * (deriv_output_wrt_diff1[i] + deriv_output_wrt_diff2[i]);
    }
}

void gaussian_weight_grad_gaussian_mean_launcher(
    torch::Tensor grad_output,
    torch::Tensor diff,
    torch::Tensor inverted_covariance,
    torch::Tensor grad_gaussian_mean
) {
    int N = diff.size(0);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    gaussian_weight_grad_gaussian_mean<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        diff.data_ptr<float>(),
        inverted_covariance.data_ptr<float>(),
        grad_gaussian_mean.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();  // Ensure the kernel execution is completed
    CUDA_CHECK(cudaGetLastError());
}


__global__ void mean_3d_to_camera_space_backward_device(
    const float* __restrict__ grad_output,
    const float* __restrict__ extrinsic_matrix,
    float* grad_mean_3d,
    int n_dims,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute the gradient: grad_mean_3d = grad_output @ extrinsic_matrix.T
    for (int i = 0; i < n_dims; ++i) {
        for (int j = 0; j < n_dims; ++j) {
            grad_mean_3d[idx * n_dims + i] +=
                grad_output[idx * n_dims + j] * extrinsic_matrix[i * n_dims + j];
        }
    }
}

void mean_3d_to_camera_space_backward_launcher(
    torch::Tensor grad_output,
    torch::Tensor extrinsic_matrix,
    torch::Tensor grad_mean_3d
) {
    int N = grad_output.size(0);
    int n_dims = grad_output.size(1);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    mean_3d_to_camera_space_backward_device<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        extrinsic_matrix.data_ptr<float>(),
        grad_mean_3d.data_ptr<float>(),
        n_dims,
        N
    );

    cudaDeviceSynchronize();  // Ensure the kernel execution is completed
    CUDA_CHECK(cudaGetLastError());
}


__global__ void camera_space_to_pixel_space_backward_device(
    const float* __restrict__ grad_output,
    const float* __restrict__ intrinsic_matrix,
    float* grad_mean_3d,
    int n_dims,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute the gradient: grad_mean_3d = grad_output @ extrinsic_matrix.T
    for (int i = 0; i < n_dims; ++i) {
        for (int j = 0; j < n_dims; ++j) {
            grad_mean_3d[idx * n_dims + i] +=
                grad_output[idx * n_dims + j] * intrinsic_matrix[i * n_dims + j];
        }
    }
}

void camera_space_to_pixel_space_backward_launcher(
    torch::Tensor grad_output,
    torch::Tensor intrinsic_matrix,
    torch::Tensor grad_mean_3d
) {
    int N = grad_output.size(0);
    int n_dims = grad_output.size(1);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    camera_space_to_pixel_space_backward_device<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        intrinsic_matrix.data_ptr<float>(),
        grad_mean_3d.data_ptr<float>(),
        n_dims,
        N
    );
}


__global__ void ndc_to_pixels_backward_device(
    const float* __restrict__ grad_output,
    const float* __restrict__ dimension,
    float* grad_ndc,
    int n_dims,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute the gradient for ndc
    grad_ndc[idx * n_dims + 0] = grad_output[idx * n_dims + 0] * (dimension[1] - 1) * 0.5f; // x-component
    grad_ndc[idx * n_dims + 1] = grad_output[idx * n_dims + 1] * (dimension[0] - 1) * 0.5f; // y-component
    grad_ndc[idx * n_dims + 2] = grad_output[idx * n_dims + 2]; // z-component
}

void ndc_to_pixels_backward_launcher(
    torch::Tensor grad_output,
    torch::Tensor dimension,
    torch::Tensor grad_ndc
) {
    int N = grad_output.size(0);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int n_dims = grad_output.size(1);

    ndc_to_pixels_backward_device<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        dimension.data_ptr<float>(),
        grad_ndc.data_ptr<float>(),
        n_dims,
        N
    );

    cudaDeviceSynchronize();  // Ensure the kernel execution is completed
    CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(gaussian_weight_derivatives, m)
{
    m.def("backward_final_color_launcher", &backward_final_color_launcher, "Backward pass for final color");
    m.def("get_alpha_backward_launcher", &get_alpha_backward_launcher, "Backward pass for alpha");
    m.def("gaussian_exp_backward_launcher", &gaussian_exp_backward_launcher, "Backward pass for gaussian exp");
    m.def("gaussian_weight_grad_inv_cov_launcher", &gaussian_weight_grad_inv_cov_launcher, "Backward pass for gaussian weight grad inv cov");
    m.def("gaussian_weight_grad_gaussian_mean_launcher", &gaussian_weight_grad_gaussian_mean_launcher, "Backward pass for gaussian weight grad gaussian mean");
    m.def("mean_3d_to_camera_space_backward_launcher", &mean_3d_to_camera_space_backward_launcher, "Backward pass for mean 3d to camera space");
    m.def("camera_space_to_pixel_space_backward_launcher", &camera_space_to_pixel_space_backward_launcher, "Backward pass for camera space to pixel space");
    m.def("ndc_to_pixels_backward_launcher", &ndc_to_pixels_backward_launcher, "Backward pass for ndc to pixels");
}
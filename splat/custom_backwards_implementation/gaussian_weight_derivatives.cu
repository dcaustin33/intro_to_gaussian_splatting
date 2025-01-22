// this will mirror the gaussian_weight_derivatives.py file
// it will consist mainly of individual __global__ functions that 
// will be stitched together in a broader backwards pass
// this will make it easier to debug and test
// goal is to test the same way I tested the custom pytorch implementation

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math_functions.h>
#include <torch/extension.h>


namespace py = pybind11;


__host__ __device__ void backward_final_color(
    const float* __restrict__ grad_output,
    const float* __restrict__ color,
    const float* __restrict__ current_T,
    const float* __restrict__ alpha,
    float* grad_color,
    float* grad_alpha,
    int idx
) {
    // Compute gradients for a single element
    grad_color[idx * 3 + 0] = grad_output[idx * 3 + 0] * current_T[idx] * alpha[idx];
    grad_color[idx * 3 + 1] = grad_output[idx * 3 + 1] * current_T[idx] * alpha[idx];
    grad_color[idx * 3 + 2] = grad_output[idx * 3 + 2] * current_T[idx] * alpha[idx];

    grad_alpha[idx] = grad_output[idx * 3 + 0] * color[idx * 3 + 0] * current_T[idx]
                    + grad_output[idx * 3 + 1] * color[idx * 3 + 1] * current_T[idx]
                    + grad_output[idx * 3 + 2] * color[idx * 3 + 2] * current_T[idx];
}

__host__ __device__ void get_alpha_backward_device(
    const float* __restrict__ grad_output,
    const float* __restrict__ gaussian_strength,
    const float* __restrict__ unactivated_opacity,
    float* grad_gaussian_strength,
    float* grad_unactivated_opacity,
    int idx
) {
    // Compute the derivative of the sigmoid function
    float sigmoid = 1.0f / (1.0f + expf(-unactivated_opacity[idx]));
    float derivative_sigmoid = sigmoid * (1.0f - sigmoid);

    // Compute gradients
    grad_gaussian_strength[idx] = grad_output[idx] * sigmoid;
    grad_unactivated_opacity[idx] = grad_output[idx] * gaussian_strength[idx] * derivative_sigmoid;
}

__host__ __device__ void gaussian_exp_backward_device(
    const float* __restrict__ grad_output,
    const float* __restrict__ gaussian_weight,
    float* grad_gaussian_weight,
    int idx
) {
    // Compute the gradient with respect to gaussian_weight
    grad_gaussian_weight[idx] = grad_output[idx] * expf(gaussian_weight[idx]);
}

__host__ __device__ void gaussian_weight_grad_inv_cov(
    const float* __restrict__ diff,
    const float* __restrict__ grad_output,
    float* grad_inv_cov,
    int batch_idx
) {
    // Compute gradient w.r.t. inverted covariance matrix: -0.5 * (diff.T @ diff)
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            grad_inv_cov[batch_idx * 4 + i * 2 + j] =
                -0.5f * grad_output[batch_idx] * diff[batch_idx * 2 + i] * diff[batch_idx * 2 + j];
        }
    }
}

__host__ __device__ void gaussian_weight_grad_gaussian_mean(
    const float* __restrict__ inverted_covariance,
    const float* __restrict__ diff,
    const float* __restrict__ grad_output,
    float* grad_gaussian_mean,
    int batch_idx
) {
    float temp[2] = {0.0f, 0.0f}; // Temporary storage for intermediate gradient calculation

    // Compute gradient w.r.t. diff: -(inv_cov @ diff + inv_cov.T @ diff)
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            temp[i] += inverted_covariance[i * 2 + j] * diff[batch_idx * 2 + j] +
                       inverted_covariance[j * 2 + i] * diff[batch_idx * 2 + j];
        }
    }

    // Scale by -0.5 * grad_output and negate to get gradient w.r.t. gaussian_mean
    for (int i = 0; i < 2; ++i) {
        grad_gaussian_mean[batch_idx * 2 + i] = -grad_output[batch_idx] * temp[i];
    }
}


__host__ __device__ void mean_3d_to_camera_space_backward_device(
    const float* __restrict__ grad_output,
    const float* __restrict__ extrinsic_matrix,
    float* grad_mean_3d,
    int batch_idx,
    int n_dims
) {
    // Initialize gradient for mean_3d to zero
    for (int i = 0; i < n_dims; ++i) {
        grad_mean_3d[batch_idx * n_dims + i] = 0.0f;
    }

    // Compute the gradient: grad_mean_3d = grad_output @ extrinsic_matrix.T
    for (int i = 0; i < n_dims; ++i) {
        for (int j = 0; j < n_dims; ++j) {
            grad_mean_3d[batch_idx * n_dims + i] +=
                grad_output[batch_idx * n_dims + j] * extrinsic_matrix[j * n_dims + i];
        }
    }
}

__host__ __device__ void camera_space_to_pixel_space_backward_device(
    const float* __restrict__ grad_output,
    const float* __restrict__ intrinsic_matrix,
    float* grad_mean_3d,
    int batch_idx,
    int n_dims
) {
    // Initialize gradient for mean_3d to zero
    for (int i = 0; i < n_dims; ++i) {
        grad_mean_3d[batch_idx * n_dims + i] = 0.0f;
    }

    // Compute the gradient: grad_mean_3d = grad_output @ intrinsic_matrix.T
    for (int i = 0; i < n_dims; ++i) {
        for (int j = 0; j < n_dims; ++j) {
            grad_mean_3d[batch_idx * n_dims + i] +=
                grad_output[batch_idx * n_dims + j] * intrinsic_matrix[j * n_dims + i];
        }
    }
}


__host__ __device__ void ndc_to_pixels_backward_device(
    const float* __restrict__ grad_output,
    const float* __restrict__ dimension,
    float* grad_ndc,
    int idx
) {
    // Compute the gradient for ndc
    grad_ndc[idx * 3 + 0] = grad_output[idx * 3 + 0] * (dimension[1] - 1) * 0.5f; // x-component
    grad_ndc[idx * 3 + 1] = grad_output[idx * 3 + 1] * (dimension[0] - 1) * 0.5f; // y-component
    grad_ndc[idx * 3 + 2] = 0.0f; // z-component gradient is zero (no effect)
}

PYBIND11_MODULE(gaussian_weight_derivatives, m)
{
    m.def("backward_final_color", &backward_final_color, "Backward pass for final color");
    m.def("get_alpha_backward_device", &get_alpha_backward_device, "Backward pass for alpha");
    m.def("gaussian_exp_backward_device", &gaussian_exp_backward_device, "Backward pass for gaussian exp");
    m.def("gaussian_weight_grad_inv_cov", &gaussian_weight_grad_inv_cov, "Backward pass for gaussian weight grad inv cov");
    m.def("gaussian_weight_grad_gaussian_mean", &gaussian_weight_grad_gaussian_mean, "Backward pass for gaussian weight grad gaussian mean");
    m.def("mean_3d_to_camera_space_backward_device", &mean_3d_to_camera_space_backward_device, "Backward pass for mean 3d to camera space");
    m.def("camera_space_to_pixel_space_backward_device", &camera_space_to_pixel_space_backward_device, "Backward pass for camera space to pixel space");
    m.def("ndc_to_pixels_backward_device", &ndc_to_pixels_backward_device, "Backward pass for ndc to pixels");
}
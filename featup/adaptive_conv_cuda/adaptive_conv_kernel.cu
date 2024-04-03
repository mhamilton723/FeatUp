#include <torch/extension.h>

#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

constexpr uint32_t kernel_channel_depth = 2;

using torch::Tensor;

template <typename scalar_t>
__launch_bounds__(1024) __global__ void adaptive_conv_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor64<scalar_t,5,torch::RestrictPtrTraits> filters,
    uint32_t batch) {

    const auto w = blockIdx.x * blockDim.x + threadIdx.x;
    const auto h = blockIdx.y * blockDim.y + threadIdx.y;
    const auto c_lo = blockIdx.z * kernel_channel_depth;
    const auto c_hi = min(c_lo + kernel_channel_depth, (uint32_t) input.size(1));

    const uint32_t I = filters.size(3);
    const uint32_t J = filters.size(4);

    if (w < out.size(3) && h < out.size(2)) {
        for (uint32_t c = c_lo; c < c_hi; c++) {
            scalar_t output_val = 0.0;
            for (uint32_t i = 0; i < I; i++) {
                for (uint32_t j = 0; j < J; j++) {

                    auto weight = filters[batch][h][w][i][j];
                    auto input_val = input[batch][c][h+i][w+j];

                    output_val += (weight * input_val);
                }
            }
            out[batch][c][h][w] = output_val;
        }
    }
}

template <typename scalar_t>
__launch_bounds__(1024) __global__ void adaptive_conv_grad_input_kernel(
    torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor64<scalar_t,5,torch::RestrictPtrTraits> filters,
    uint32_t batch) {

    const int32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t h = blockIdx.y * blockDim.y + threadIdx.y;

    const int32_t H_out = out.size(2);
    const int32_t W_out = out.size(3);

    // thread's output index is outside output tensor
    if (w >= W_out || h >= H_out) return;

    const int32_t c_lo = blockIdx.z * kernel_channel_depth;
    const int32_t c_hi = min(c_lo + kernel_channel_depth, (int32_t) out.size(1));

    const int32_t I = filters.size(3);
    const int32_t J = filters.size(4);

    const int32_t H_grad = grad_output.size(2);
    const int32_t W_grad = grad_output.size(3);

    for (int32_t c = c_lo; c < c_hi; c++) {

        scalar_t output_val = 0.0;

        for (int32_t i = 0; i < I; i++) {
            for (int32_t j = 0; j < J; j++) {
                const int32_t h_grad = h - i;
                const int32_t w_grad = w - j;

                if (h_grad >= 0 && w_grad >= 0 && h_grad < H_grad && w_grad < W_grad) {
                    output_val += grad_output[batch][c][h_grad][w_grad] * filters[batch][h_grad][w_grad][i][j];
                }
            }
        }
        out[batch][c][h][w] = output_val;
    }
}


template <typename scalar_t>
__launch_bounds__(1024) __global__ void adaptive_conv_grad_filters_kernel(
    torch::PackedTensorAccessor64<scalar_t,5,torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> input,
    uint32_t batch) {

    const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t h = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t f = blockIdx.z * blockIdx.z + threadIdx.z;

    const uint32_t H = out.size(1);
    const uint32_t W = out.size(2);
    const uint32_t I = out.size(3);
    const uint32_t J = out.size(4);

    assert(I == J);

    const uint32_t C = input.size(1);

    if (h >= H || w >= W || f >= (I * J)) return;
    
    const uint32_t i = f / I;
    const uint32_t j = f % I;

    scalar_t output_val = 0.0;
    for (uint32_t c = 0; c < C; c++) {
        auto grad = grad_output[batch][c][h][w];
        auto input_val = input[batch][c][h+i][w+j];
        output_val += grad * input_val;
    }
    out[batch][h][w][i][j] = output_val;
}


template <typename T>
T div_round_up(T a, T b) {
    return (a + b - 1) / b;
}

Tensor adaptive_conv_cuda_forward(Tensor input, Tensor filters) {
    at::cuda::set_device(input.device().index());

    // Check for error in the input tensors
    TORCH_CHECK(input.dim() == 4, "input must have 4 dimensions");
    TORCH_CHECK(filters.dim() == 5, "filters must have 5 dimensions");
    TORCH_CHECK(input.dtype() == filters.dtype(), "input and filters must have the same data type");

    const uint32_t B = input.size(0);
    const uint32_t C = input.size(1);
    const uint32_t H_in = input.size(2);
    const uint32_t W_in = input.size(3);

    TORCH_CHECK(filters.size(0) == B, "Inconsistent batch size between input and filters");
    const uint32_t H_out = filters.size(1);
    const uint32_t W_out = filters.size(2);
    const uint32_t I = filters.size(3);
    const uint32_t J = filters.size(4);

    TORCH_CHECK(I == J, "filters dimension I and J must be equal");
    TORCH_CHECK(H_out + I - 1 == H_in, "Inconsistent height between input and filters");
    TORCH_CHECK(W_out + J - 1 == W_in, "Inconsistent width between input and filters");

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(torch::kCUDA);

    auto out = torch::zeros({ B, C, H_out, W_out }, options);

    const dim3 tpb(32, 32);
    const dim3 blocks(div_round_up(W_out, tpb.x),
                      div_round_up(H_out, tpb.y),
                      div_round_up(C, kernel_channel_depth));

    for (uint32_t b = 0; b < B; b++) {
        AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "adaptive_conv_forward_cuda", ([&] {
            adaptive_conv_forward_kernel<scalar_t><<<blocks,tpb>>>(
                out.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                input.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                filters.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
                b);
            }));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in adaptive_conv_forward_kernel: %s\n", cudaGetErrorString(err));
        }
    }
    return out;
}


Tensor adaptive_conv_cuda_grad_input(Tensor grad_output, Tensor filters) {
    at::cuda::set_device(grad_output.device().index());

    // Check for error in the input tensors
    TORCH_CHECK(grad_output.dim() == 4, "grad_output must have 4 dimensions");
    TORCH_CHECK(filters.dim() == 5, "filters must have 5 dimensions");

    const uint32_t B = grad_output.size(0);
    const uint32_t C = grad_output.size(1);
    const uint32_t H_out = grad_output.size(2);
    const uint32_t W_out = grad_output.size(3);

    TORCH_CHECK(filters.size(0) == B, "Inconsistent batch size between filters and grad_output");
    TORCH_CHECK(filters.size(1) == H_out, "Inconsistent height between filters and grad_output");
    TORCH_CHECK(filters.size(2) == W_out, "Inconsistent width between filters and grad_output");

    const uint32_t I = filters.size(3);
    const uint32_t J = filters.size(4);
    TORCH_CHECK(I == J, "filters dimension I and J must be equal");

    const uint32_t H_in = H_out + I - 1;
    const uint32_t W_in = W_out + J - 1;

    TORCH_CHECK(grad_output.dtype() == filters.dtype(), "grad_output and filters must have the same data type");

    auto options = torch::TensorOptions()
        .dtype(filters.dtype())
        .device(torch::kCUDA);

    auto out = torch::zeros({ B, C, H_in, W_in }, options);

    const dim3 tpb(32, 32);
    const dim3 blocks(div_round_up(W_in, tpb.x),
                      div_round_up(H_in, tpb.y),
                      div_round_up(C, kernel_channel_depth));

    for (uint32_t b = 0; b < B; b++) {
        AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "adaptive_conv_grad_input_cuda", ([&] {
            adaptive_conv_grad_input_kernel<scalar_t><<<blocks,tpb>>>(
                out.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                grad_output.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                filters.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
                b);
            }));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in adaptive_conv_grad_input_kernel: %s\n", cudaGetErrorString(err));
        }
    }
    return out;
}

Tensor adaptive_conv_cuda_grad_filters(Tensor grad_output, Tensor input) {
    at::cuda::set_device(grad_output.device().index());

    // Check for error in the input tensors
    TORCH_CHECK(grad_output.dim() == 4, "grad_output must have 4 dimensions");
    TORCH_CHECK(input.dim() == 4, "input must have 4 dimensions");

    const uint32_t B = grad_output.size(0);
    const uint32_t C = grad_output.size(1);
    const uint32_t H_out = grad_output.size(2);
    const uint32_t W_out = grad_output.size(3);

    TORCH_CHECK(input.size(0) == B, "Inconsistent batch size between input and grad_output");
    TORCH_CHECK(input.size(1) == C, "Inconsistent number of channels between input and grad_output");

    const uint32_t H_in = input.size(2);
    const uint32_t W_in = input.size(3);

    TORCH_CHECK(H_in > H_out, "Input height must be greater than grad_output height");
    TORCH_CHECK(W_in > W_out, "Input width must be greater than grad_output width");

    const uint32_t I = W_in - W_out + 1;
    const uint32_t J = H_in - H_out + 1;

    TORCH_CHECK(grad_output.dtype() == input.dtype(), "grad_output and input must have the same data type");

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(torch::kCUDA);

    auto out = torch::zeros({ B, H_out, W_out, I, J }, options);

    const dim3 tpb(32, 32, 1);
    const dim3 blocks(div_round_up(W_out, tpb.x),
                      div_round_up(H_out, tpb.y),
                      div_round_up(I * J, tpb.z));



    for (uint32_t b = 0; b < B; b++) {
        AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "adaptive_conv_grad_filters_cuda", ([&] {
            adaptive_conv_grad_filters_kernel<scalar_t><<<blocks,tpb>>>(
                out.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
                grad_output.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                input.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                b);
            }));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in adaptive_conv_grad_filters_kernel: %s\n", cudaGetErrorString(err));
        }
    }
    return out;
}


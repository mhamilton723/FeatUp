#include <torch/extension.h>
using torch::Tensor;

// CUDA forward declarations

Tensor adaptive_conv_cuda_forward(Tensor input, Tensor filters);
Tensor adaptive_conv_cuda_grad_input(Tensor grad_output, Tensor filters);
Tensor adaptive_conv_cuda_grad_filters(Tensor grad_output, Tensor input);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

Tensor adaptive_conv_forward(Tensor input, Tensor filters) {
    //CHECK_INPUT(input);
    //CHECK_INPUT(filters);
    return adaptive_conv_cuda_forward(input, filters);
}

Tensor adaptive_conv_grad_input(Tensor grad_output, Tensor filters) {
    //CHECK_INPUT(grad_output);
    //CHECK_INPUT(filters);
    return adaptive_conv_cuda_grad_input(grad_output, filters);
}

Tensor adaptive_conv_grad_filters(Tensor grad_output, Tensor input) {
    //CHECK_INPUT(grad_output);
    //CHECK_INPUT(input);
    return adaptive_conv_cuda_grad_filters(grad_output, input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_conv_forward, "adaptive_conv forward");
    m.def("grad_input", &adaptive_conv_grad_input, "adaptive_conv grad_input");
    m.def("grad_filters", &adaptive_conv_grad_filters, "adaptive_conv grad_filters");
}

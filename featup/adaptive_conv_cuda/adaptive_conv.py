from torch.autograd import Function
import torch

from featup.adaptive_conv_cuda import cuda_impl

torch.manual_seed(42)


class AdaptiveConv(Function):

    @staticmethod
    def forward(ctx, input, filters):

        ctx.save_for_backward(filters, input)

        b, c_in, h, w = input.shape
        b, h2, w2, f1, f2 = filters.shape

        assert f1 == f2
        kernel_size = f1

        result = cuda_impl.forward(input, filters)
        return result

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.contiguous()
        
        filters, input = ctx.saved_tensors
        grad_input = grad_filters = None
        b, c, h1, w1 = input.shape
        b, h2, w2, f1, f2 = filters.shape

        assert grad_output.is_cuda
        assert input.is_cuda
        assert filters.is_cuda

        assert f1 == f2
        kernel_size = f1

        if ctx.needs_input_grad[0]:
            grad_input = cuda_impl.grad_input(grad_output, filters)
        
        if ctx.needs_input_grad[1]:
            grad_filters = cuda_impl.grad_filters(grad_output, input)
            
        return grad_input, grad_filters

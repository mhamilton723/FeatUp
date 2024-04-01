#include <cassert>

#include <torch/extension.h>
using torch::Tensor;

Tensor adaptive_conv_forward(Tensor input, Tensor filters) {

    assert(input.dtype() == filters.dtype());

    auto B = input.sizes()[0];
    auto C_in = input.sizes()[1];
    auto H_in = input.sizes()[2];
    auto W_in = input.sizes()[3];

    assert(filters.sizes()[0] == B);
    auto H_out = filters.sizes()[1];
    auto W_out = filters.sizes()[2];
    auto I = filters.sizes()[3];
    auto J = filters.sizes()[4];

    assert(I == J);
    assert(H_out + I - 1 == H_in);
    assert(W_out + J - 1 == W_in);

    auto out = torch::zeros({ B, C_in, H_out, W_out }, input.dtype());
    
    // output stationary
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t c = 0; c < C_in; c++) {
            for (uint32_t h = 0; h < H_out; h++) {
                for (uint32_t w = 0; w < W_out; w++) {
                    // produce output pixel b, h, w, c
                    for (uint32_t i = 0; i < I; i++) {
                        for (uint32_t j = 0; j < J; j++) {
                            auto weight = filters[b][h][w][i][j];
                            assert(h+i < H_in);
                            assert(w+j < W_in);
                            auto input_val = input[b][c][h+i][w+j];
                            out[b][c][h][w] += weight * input_val;
                        }
                    }
                }
            }
        }
    }
    return out;
}

Tensor adaptive_conv_grad_input(Tensor grad_output, Tensor filters) {
   
    auto B = grad_output.sizes()[0];
    auto C = grad_output.sizes()[1];
    auto H_out = grad_output.sizes()[2];
    auto W_out = grad_output.sizes()[3];

    assert(filters.sizes()[0] == B);
    assert(filters.sizes()[1] == H_out);
    assert(filters.sizes()[2] == W_out);
    auto I = filters.sizes()[3];
    auto J = filters.sizes()[4];
    assert(I == J);

    auto H_in = H_out + I - 1;
    auto W_in = W_out + J - 1;

    assert(grad_output.dtype() == filters.dtype());

    auto out = torch::zeros({ B, C, H_in, W_in }, grad_output.dtype());

    for (int32_t b = 0; b < B; b++) {
        for (int32_t c = 0; c < C; c++) {
            for (int32_t h = 0; h < H_in; h++) {
                for (int32_t w = 0; w < W_in; w++) {
                    for (int32_t i = 0; i < I; i++) {
                        for (int32_t j = 0; j < J; j++) {
                            
                            int32_t h_out = h - i;
                            int32_t w_out = w - j;

                            if ((h_out >= 0) && (w_out >= 0) && (h_out < H_out) && (w_out < W_out)) {
                                auto grad = grad_output[b][c][h_out][w_out];
                                auto weight = filters[b][h_out][w_out][i][j];

                                out[b][c][h][w] += grad * weight;
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

Tensor adaptive_conv_grad_filters(Tensor grad_output, Tensor input) {

    auto B = grad_output.sizes()[0];
    auto C = grad_output.sizes()[1];
    auto H_out = grad_output.sizes()[2];
    auto W_out = grad_output.sizes()[3];

    assert(input.sizes()[0] == B);
    assert(input.sizes()[1] == C);
    auto H_in = input.sizes()[2];
    auto W_in = input.sizes()[3];

    assert(H_in > H_out);
    assert(W_in > W_out);

    auto I = W_in - W_out + 1;
    auto J = H_in - H_out + 1;
    
    assert(grad_output.dtype() == input.dtype());

    auto out = torch::zeros({ B, H_out, W_out, I, J }, grad_output.dtype());

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < H_out; h++) {
            for (uint32_t w = 0; w < W_out; w++) {
                for (uint32_t i = 0; i < I; i++) {
                    for (uint32_t j = 0; j < J; j++) {
                        for (uint32_t c = 0; c < C; c++) {
                            auto grad = grad_output[b][c][h][w];
                            assert(h + i < H_in);
                            assert(w + j < W_in);
                            auto input_val = input[b][c][h+i][w+j];
                            out[b][h][w][i][j] += grad * input_val;
                        }
                    }
                }
            }
        }
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_conv_forward, "adaptive_conv forward");
    m.def("grad_input", &adaptive_conv_grad_input, "adaptive_conv grad_input");
    m.def("grad_filters", &adaptive_conv_grad_filters, "adaptive_conv grad_filters");
}

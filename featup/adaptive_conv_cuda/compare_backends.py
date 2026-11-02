"""Compare the AdaptiveConv backends: C++ (CPU), CUDA kernel (GPU) and plain PyTorch (GPU).

Plain PyTorch uses autograd, so it is the reference for gradients.

Usage: python featup/adaptive_conv_cuda/compare_backends.py
"""
import time

import torch

from featup.adaptive_conv_cuda.adaptive_conv import AdaptiveConv

B, C, H, W, K = 2, 5, 37, 41, 7  # non-square, and crosses the kernel's 32x32 / 2-channel blocks
TOLERANCE = 1e-10
SUSPECT_POSITIONS = {(0, 0), (0, 1), (0, 4), (1, 2), (2, 2), (3, 4), (5, 1)}


def plain_adaptive_conv(input, filters):
    _, h, w, ki, kj = filters.shape
    out = 0
    for i in range(ki):
        for j in range(kj):
            out = out + filters[:, :, :, i, j].unsqueeze(1) * input[:, :, i:i + h, j:j + w]
    return out


def compiled_adaptive_conv(input, filters):
    return AdaptiveConv.apply(input, filters)


def run(fn, input, filters, weights, device):
    input = input.to(device, copy=True).requires_grad_()
    filters = filters.to(device, copy=True).requires_grad_()
    out = fn(input, filters)
    (out * weights.to(device)).sum().backward()
    return {
        "output": out.detach().cpu(),
        "grad_input": input.grad.cpu(),
        "grad_filters": filters.grad.cpu(),
    }


def compare(name, result, reference):
    print(f"\n{name} vs plain")
    passed = True
    for key in reference:
        diff = (result[key] - reference[key]).abs().max().item()
        ok = diff <= TOLERANCE
        passed = passed and ok
        print(f"  {key:<13} max abs diff {diff:.3e}  {'PASS' if ok else 'FAIL'}")
    return passed


def print_filter_positions(name, grad_filters):
    nonzero = grad_filters.abs().sum((0, 1, 2)) > 0
    print(f"\n{name} grad_filters nonzero kernel positions ({nonzero.sum().item()} of {K * K}):")
    for row in nonzero.tolist():
        print("  " + " ".join("X" if v else "." for v in row))
    return {tuple(p) for p in nonzero.nonzero().tolist()}


def compare_gradients():
    torch.manual_seed(0)
    input = torch.randn(B, C, H + K - 1, W + K - 1, dtype=torch.float64)
    filters = torch.randn(B, H, W, K, K, dtype=torch.float64)
    weights = torch.randn(B, C, H, W, dtype=torch.float64)

    reference = run(plain_adaptive_conv, input, filters, weights, "cuda")
    cpp = run(compiled_adaptive_conv, input, filters, weights, "cpu")
    cuda = run(compiled_adaptive_conv, input, filters, weights, "cuda")

    cpp_ok = compare("cpp", cpp, reference)
    cuda_ok = compare("cuda", cuda, reference)

    cuda_positions = print_filter_positions("cuda", cuda["grad_filters"])
    print_filter_positions("plain", reference["grad_filters"])

    mask = torch.zeros(K, K, dtype=torch.bool)
    for i, j in cuda_positions:
        mask[i, j] = True
    kept_diff = (cuda["grad_filters"] - reference["grad_filters"])[..., mask].abs().max().item()
    print(f"\ncuda vs plain at cuda's nonzero positions: max abs diff {kept_diff:.3e}")
    print(f"cuda positions equal the suspected squares pattern: {cuda_positions == SUSPECT_POSITIONS}")
    print(f"\nAll cpp checks passed: {cpp_ok}")
    print(f"All cuda checks passed: {cuda_ok}")


def time_forward(fn, input, filters, n_warmup=5, n_runs=20):
    with torch.no_grad():
        for _ in range(n_warmup):
            fn(input, filters)
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn(input, filters)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return sum(times) / len(times) * 1000


def compare_speed():
    torch.manual_seed(0)
    input = torch.randn(1, 384, 224 + K - 1, 224 + K - 1, device="cuda")
    filters = torch.randn(1, 224, 224, K, K, device="cuda")
    print(f"\nForward time at real size ({torch.cuda.get_device_name()}, float32, mean of 20 runs)")
    print(f"  cuda   {time_forward(compiled_adaptive_conv, input, filters):8.2f} ms")
    print(f"  plain  {time_forward(plain_adaptive_conv, input, filters):8.2f} ms")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, shapes B={B} C={C} H={H} W={W} K={K}, float64")
    compare_gradients()
    compare_speed()

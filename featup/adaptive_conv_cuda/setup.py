from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='adaptive_conv_cuda',
    ext_modules=[
        CUDAExtension('cuda_impl', [
            'adaptive_conv_cuda.cpp',
            'adaptive_conv_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

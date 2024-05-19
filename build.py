from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup(
        ext_modules=[
            CUDAExtension(
                'adaptive_conv_cuda_impl',
                [
                    'featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp',
                    'featup/adaptive_conv_cuda/adaptive_conv_kernel.cu',
                ]),
            CppExtension(
                'adaptive_conv_cpp_impl',
                ['featup/adaptive_conv_cuda/adaptive_conv.cpp'],
                undef_macros=["NDEBUG"]),
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )

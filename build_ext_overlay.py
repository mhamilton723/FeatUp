from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
overlay = [os.path.join(_HERE,"ext_build/overlay")]
setup(
    name="featup-ext-test",
    ext_modules=[
        CUDAExtension("adaptive_conv_cuda_impl",
                      ["featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp",
                       "featup/adaptive_conv_cuda/adaptive_conv_kernel.cu"],
                       include_dirs=overlay),
        CppExtension("adaptive_conv_cpp_impl",
                     ["featup/adaptive_conv_cuda/adaptive_conv.cpp"],
                     undef_macros=["NDEBUG"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)

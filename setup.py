from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='featup',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'torch',
        'kornia',
        'omegaconf',
        'pytorch-lightning',
        'torchvision',
        'tqdm',
        'torchmetrics',
        'scikit-learn',
        'numpy',
        'matplotlib',
        'timm==0.4.12',
    ],
    author='Mark Hamilton, Stephanie Fu',
    author_email='markth@mit.edu, fus@berkeley.edu',
    description='Official code for "FeatUp: A Model-Agnostic Frameworkfor Features at Any Resolution" ICLR 2024',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mhamilton723/FeatUp',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
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

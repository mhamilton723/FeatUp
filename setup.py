from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='featup',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'kornia',
        'omegaconf',
        'pytorch-lightning',
        'torchvision',
        'tqdm',
        'torchemetrics',
        'sklearn',
        'numpy',
        'matplotlib',
        'glob',

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
        CUDAExtension('featup.adaptive_conv_cuda.cuda_impl', [
            'featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp',
            'featup/adaptive_conv_cuda/adaptive_conv_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

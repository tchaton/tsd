import platform
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

extra_compile_args = []
if platform.system() != 'Windows':
    extra_compile_args += ['-Wno-unused-variable']

if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    extra_compile_args += ['-DVERSION_GE_1_3']

ext_modules = [
    CppExtension('tsd.scatter_cpu', ['cpu/scatter.cpp'],
                 extra_compile_args=extra_compile_args),

    CppExtension('tsd.hag_utils', ['hag_utils/hag.cpp', 'hag_utils/gnn_to_hag.cpp'],
                 extra_compile_args=extra_compile_args)
]
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

if CUDA_HOME is not None:
    ext_modules += [
        CUDAExtension('tsd.scatter_cuda',
                      ['cuda/scatter.cpp', 'cuda/scatter_kernel.cu'])
    ]

__version__ = '0.0.1'

install_requires = []
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='tsd',
    version=__version__,
    description='PyTorch Extension Library of Optimized Scatter Operations',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    keywords=[
        'pytorch',
        'scatter',
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)

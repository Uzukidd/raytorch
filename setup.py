import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources],
    )
    
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.1.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'raytorch/version.py')

    setup(
        name='raytorch',
        version=version,
        description='Raytorch: a differentiable ray intersection using Pytorch',
        author='Uzuki Ishikawajima',
        author_email='uzukidd@gmail.com',
        license='MIT License',
        packages=find_packages(exclude=['test']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
           
        ],
    )

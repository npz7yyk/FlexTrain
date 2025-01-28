# Parts of the code here are adapted from DeepSpeed
#
# Repository: https://github.com/microsoft/DeepSpeed
# File: setup.py
# Commit: 0d9cfa0
# License: Apache-2.0

"""
FlexTrain library
"""

import os
import subprocess
from setuptools import setup, find_packages
import time
import typing

torch_available = True
accelerator_name = "cuda"
try:
    import torch
except ImportError:
    raise ImportError("Please install torch before running setup.py")

RED_START = '\033[31m'
RED_END = '\033[0m'
ERROR = f"{RED_START} [ERROR] {RED_END}"


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def is_env_set(key):
    """
    Checks if an environment variable is set and not "".
    """
    return bool(os.environ.get(key, None))


def get_env_if_set(key, default: typing.Any = ""):
    """
    Returns an environment variable if it is set and not "",
    otherwise returns a default value. In contrast, the fallback
    parameter of os.environ.get() is skipped if the variable is set to "".
    """
    return os.environ.get(key, None) or default


install_requires = fetch_requirements('requirements.txt')

if torch_available:
    TORCH_MAJOR = torch.__version__.split('.')[0]
    TORCH_MINOR = torch.__version__.split('.')[1]
else:
    TORCH_MAJOR = "0"
    TORCH_MINOR = "0"

# We rely on JIT on Linux, opposite on Windows.
BUILD_OP_PLATFORM = 0
BUILD_OP_DEFAULT = 0
print(f"BUILD_OPS={BUILD_OP_DEFAULT}")


def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


# Write out version/git info.
git_hash_cmd = "git rev-parse --short HEAD"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
if command_exists('git') and not is_env_set('BUILD_STRING'):
    try:
        result = subprocess.check_output(git_hash_cmd, shell=True)
        git_hash = result.decode('utf-8').strip()
        result = subprocess.check_output(git_branch_cmd, shell=True)
        git_branch = result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"
        git_branch = "unknown"
else:
    git_hash = "unknown"
    git_branch = "unknown"


def create_dir_symlink(src, dest):
    if not os.path.islink(dest):
        if os.path.exists(dest):
            os.remove(dest)
        assert not os.path.exists(dest)
        os.symlink(src, dest)


# Parse the FlexTrain version string from version.txt.
version_str = open('version.txt', 'r').read().strip()

# Build specifiers like .devX can be added at install time.
# Otherwise, add the git hash.
# Example: BUILD_STRING=".dev20201022" python setup.py sdist bdist_wheel.

# Building wheel for distribution, update version file.
if is_env_set('BUILD_STRING'):
    # Build string env specified, probably building for distribution.
    with open('build.txt', 'w') as fd:
        fd.write(os.environ['BUILD_STRING'])
    version_str += os.environ['BUILD_STRING']
elif os.path.isfile('build.txt'):
    # build.txt exists, probably installing from distribution.
    with open('build.txt', 'r') as fd:
        version_str += fd.read().strip()
else:
    # None of the above, probably installing from source.
    version_str += f'+{git_hash}'

torch_version = ".".join([TORCH_MAJOR, TORCH_MINOR])
bf16_support = False
# Set cuda_version to 0.0 if cpu-only.
cuda_version = "0.0"
nccl_version = "0.0"
if torch_available and torch.version.cuda is not None:
    cuda_version = ".".join(torch.version.cuda.split('.')[:2])
    if isinstance(torch.cuda.nccl.version(), int):
        # This will break if minor version > 9.
        nccl_version = ".".join(str(torch.cuda.nccl.version())[:2])
    else:
        nccl_version = ".".join(map(str, torch.cuda.nccl.version()[:2]))
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_available():
        bf16_support = torch.cuda.is_bf16_supported()
torch_info = {
    "version": torch_version,
    "bf16_support": bf16_support,
    "cuda_version": cuda_version,
    "nccl_version": nccl_version
}

print(f"version={version_str}, git_hash={git_hash}, git_branch={git_branch}")
with open('flextrain/git_version_info_installed.py', 'w') as fd:
    fd.write(f"version='{version_str}'\n")
    fd.write(f"git_hash='{git_hash}'\n")
    fd.write(f"git_branch='{git_branch}'\n")
    fd.write(f"accelerator_name='{accelerator_name}'\n")
    fd.write(f"torch_info={torch_info}\n")

print(f'install_requires={install_requires}')

# Parse README.md to make long_description for PyPI page.
thisdir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(thisdir, 'README.md'), encoding='utf-8') as fin:
    readme_text = fin.read()

start_time = time.time()

setup(
    name='flextrain',
    version=version_str,
    description='FlexTrain library',
    long_description=readme_text,
    long_description_content_type='text/markdown',
    project_urls={
        'Source': 'https://github.com/npz7yyk/FlexTrain',
    },
    install_requires=install_requires,
    packages=find_packages(include=['flextrain', 'flextrain.*']),
    include_package_data=True,
    license='Apache Software License 2.0'
)

end_time = time.time()
print(f'flextrain build time = {end_time - start_time} secs')

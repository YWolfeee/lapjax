# MIT License

# Copyright (c) 2023 Haotian Ye

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================
"""Setup for pip package."""

from setuptools import find_packages
from setuptools import setup
import os
import shutil
try:
  import jax
except:
  raise ImportError('`jax` package is required for setup.')

REQUIRED_PACKAGES = [
  'absl-py>=1.4.0',
  'attrs>=21.2.0',
  'h5py>=3.8.0',
  'chex>=0.1.5',
  'optax>=0.1.4',
  'numpy>=1.21.5',
  'scipy>=1.7.3',
  'typing_extensions>=4.5.0',
  'dm-haiku>=0.0.9',
  'jax>=0.3.7',
]

with open("README.md", "r") as fh:
  long_description = fh.read()

file_content = \
"import sys, importlib\n" + \
"from lapjax.lapsrc.wrapper import _wrap_module\n" + \
"_wrap_module(importlib.import_module(__name__.replace('lapjax', 'jax')), \n" + \
"             sys.modules[__name__])\n"

def create_py(dest: os.path, src: os.path, pkg_name: str):
  """
  For all python files in src, copy the file to dest and change the content
  to import the file in source. Create `.pyi` files for all `.py` files.
  """

  for filename in os.listdir(src):
    srcpath = os.path.join(src, filename)
    destpath = os.path.join(dest, filename)
    if os.path.isdir(srcpath) and filename != '__pycache__':
      print(f'Calling create_py for {pkg_name}.{filename} recursively.')
      if not os.path.exists(destpath):
        os.mkdir(destpath)
      create_py(os.path.join(dest, filename), srcpath, f'{pkg_name}.{filename}')

    elif filename == '__init__.py':  # python init file
      if os.path.exists(destpath):
        continue
      with open(destpath, 'w') as f:
        for pkg in os.listdir(src):
          if pkg.endswith('.py') and pkg not in ['__init__.py', 'iree.py']:
            # import from wrapped module
            f.write(f'from lap{pkg_name} import {pkg[:-3]} as {pkg[:-3]}\n')
        f.write(file_content)
      with open(destpath + 'i', 'w') as f:
        f.write(f'from {pkg_name} import *')

    elif filename.endswith('.py'):   # standard python file      
      with open(destpath, 'w') as f:
        f.write(file_content)
      with open(destpath + 'i', 'w') as f:
        f.write(f'from {pkg_name}.{filename[:-3]} import *')


def pre_setup():
  """Pre-setup function. Clean the directory.
  Change the `_lapsrc` directory to `lapjax` that can be used for setup.
  Includes the `jax` package structure.
  """
  assert os.path.exists('_lapsrc'), \
    "Please run setup.py in the root directory of lapjax."
  shutil.rmtree('build', ignore_errors=True) 
  shutil.rmtree('lapjax.egg-info', ignore_errors=True) 
  # Remove the old `lapjax` directory.
  if os.path.exists('lapjax'):
    shutil.rmtree('lapjax')
  # Copy the `_lapsrc` directory to `lapjax`.
  os.mkdir('lapjax')
  shutil.copytree('_lapsrc', 'lapjax/lapsrc')
  shutil.move('lapjax/lapsrc/__init__.py', 'lapjax/__init__.py')
  shutil.move('lapjax/lapsrc/__init__.pyi', 'lapjax/__init__.pyi')
  f = open('lapjax/lapsrc/__init__.py', 'x')
  f.close()
  # os.mknod("lapjax/lapsrc/__init__.py")
  # Copy the `jax` package structure to `lapjax`.
  create_py('lapjax', jax.__path__[0], 'jax')
pre_setup()

setup(
  name='lapjax',
  version='0.0',
  author='Haotian Ye',
  author_email='',
  description='A package for computing the laplacian automatically '
              'using a technique budded "Forward Laplacian".',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/YWolfeee/lapjax',
  packages=find_packages(exclude=['_lapsrc']),
  include_package_data=True,
  install_requires=REQUIRED_PACKAGES,
  extras_require={'testing': ['flake8', 'pylint', 'pytest', 'pytype']},
  platforms=['any'],
  license='MIT',
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
  ],
)

def post_setup():
  """Post-setup function.
  Remove the `lapjax` directory.
  """
  shutil.rmtree('lapjax')
post_setup()
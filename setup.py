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


REQUIRED_PACKAGES = [
    'absl-py==1.4.0',
    'attrs==21.2.0',
    'h5py==3.8.0',
    'chex==0.1.5',
    'optax==0.1.4',
    'numpy==1.21.5',
    'scipy==1.7.3',
    'typing_extensions==4.5.0',
    'dm-haiku==0.0.9',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

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
    packages=find_packages(),
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

## LapJAX: Automatic forward-mode laplacian computation JAX
##### WIP Repo
LapJAX is a package built upon Google JAX. The purpose of this package is to compute the laplacian automatically using a technique budded "Forward Laplacian". The mathematical introduction can be founded [here](#forward-laplacian-introduction).


### Installation
To install LapJAX together with all the dependencies (excluding JAX), go to the downloaded directory and run

```shell
pip install .
```

Currently we do not support -e option (editable installation). If you have a GPU available, then you can
install JAX with CUDA support, using e.g.:

```shell
pip3 install --upgrade jax[cuda]==0.3.24 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Note that the jaxlib version must correspond to the existing CUDA installation
you wish to use. Please see the
[JAX documentation](https://github.com/google/jax#installation) for more
details.

### Usage
Assume you have written a function `f(x)` using jax, and you want to compute
  the laplacian of `f(x)` w.r.t. `x`.
To use lapjax, take the following steps:
1. Replace `jax` with `lapjax`, e.g.,
```python
# import jax.numpy as jnp
# from jax import vmap
import lapjax.numpy as jnp
from lapjax import vmap

def f(x):
  # Your code here.
  pass
```
or use a sys.modules replacement trick (not recommended):
```python
import lapjax
import sys
sys.modules['jax'] = lapjax
```
2. Creat a LapTuple from the input, pass it to the function directly, and obtain the laplacian, e.g.
```python
from lapjax import LapTuple, TupType

# input_x is the input of f(x)
def lap_of_f(x):
  input_x = LapTuple(input_x, is_input = True)
  output = f(input_x)
  return output.get(TupType.LAP)
```
That's it. All you need to do is to ensure that inside function f(x), all calls to JAX functions are passed to LapJAX.
### Function Support
LapJax encapsulates many commonly used JAX functions (most of which are in `jax.numpy`), such that they can take LapTuple as input (rather than `ndarray` only). However, we acknowledge that the supported functions are a subset of JAX functions. To see the complete supported function lists, see [Supported Functions](#supported-functions); To add new functions for your code, see [Add Your Functions](#add-your-functions).

### Forward Laplacian Introduction
To be implemented.
### Supported Functions
Ideally, we should support all JAX functions where the input can be a JAX `ndarray`. However, the current implementation only supports a few of them. We split currently supported functions into several types, depending on their semantic and the way we accelerate them.
#### Construction Functions
These include functions where the output should not depend on the value of potential LapTuple inputs, i.e. $\Delta f(x) = 0$. We ensure that the call to these functions is normal when the inputs contain LapTuple. Specifically, they are
```python
jax.numpy.eye, jax.numpy.ones, jax.numpy.ones_like, jax.numpy.zeros, jax.numpy.zeros_like, jax.numpy.asarray, jax.numpy.sign, jax.lax.stop_gradient
```
#### Element-wise Functions
These include functions where the output depends element-wisely on the input, and the shape remains unchanged. We leverage this sparsity when computing the laplacian. Specifically, they are
```python
jax.numpy.sin, jax.numpy.cos, jax.numpy.tan, jax.numpy.sinh, jax.numpy.cosh, jax.numpy.tanh, jax.numpy.exp, jax.numpy.log, jax.numpy.square, jax.numpy.sqrt, jax.numpy.power, jax.numpy.abs,
```
#### Linear Functions
These include functions where we can directly apply functions separately on the input gradient and laplacian to obtain the output gradient and laplacian, i.e., the transformation of value, gradient, and laplacian are identical to `f`. Specifically, they are
```python
jax.numpy.reshape, jax.numpy.transpose, jax.numpy.swapaxes, jax.numpy.split, jax.numpy.array_split, jax.numpy.concatenate, jax.numpy.squeeze, jax.numpy.expand_dims, jax.numpy.repeat, jax.numpy.tile, jax.numpy.triu, jax.numpy.tril, jax.numpy.sum, jax.numpy.mean,
```
#### Overload Functions
These inlcude functions w.r.t. python operator, such as `+` and `-`. Specifically, they are
```python
jax.numpy.add, jax.numpy.subtract, jax.numpy.multiply, jax.numpy.divide, jax.numpy.true_divide,
```
#### Customized Functions
These are functions where we customize the gradient and laplacian computation to accelerate. We need to right their gradient and laplacian functions manually. Specifically, they are
```python
jax.numpy.matmul, jax.numpy.dot, jax.numpy.max (amax), jax.numpy.min (amin), jax.numpy.linalg.slogdet, jax.nn.logsumexp, jax.nn.softmax,
```
#### Merging Functions
These include functions where the input has a single ndarray, and operations are taken along several axes. Functions that do not belong to previous types and that we do not have a better way to compute can be placed here. However, as we directly calculate the hessian to obtain the laplacian, functions here might result in an OOM issue, especially when they involve operations on a huge matrix. Specifically, they are
```python
jax.numpy.linalg.norm, jax.numpy.prod,
```
#### Auxiliary Functions
These are JAX relevant auxiliary functions. We support `jax.vmap` with LapTuple as vectorized inputs.
### Add Your Functions
To add new functions, please follow the steps below:
1. Check whether the function can be classified as an existing function type listed above. For example, for construction function, linear function and element-wise function, you can simply wrap them in the package (e.g. `lapjax/numpy.py`), add them to the `function_list` in corresponding function class (see `function_class.py`).
2. If the function is special (you need to customize personal derivative computation and laplacian computation), please add them in the customized function class, and write your personal operations inside.
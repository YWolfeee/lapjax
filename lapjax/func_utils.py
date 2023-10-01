"""This module includes functional utility tools.
Tools here should not depend on the class in 
  `laptuple.py` and `sparsinfo.py`.
"""
from functools import partial
from typing import Sequence, Union, Callable, Tuple

import jax
import jax.numpy as jnp
from lapjax.lapconfig import lapconfig


def lap_print(*args, **kwargs):
  if lapconfig.debug_print:
    print(*args, **kwargs)

# curry and wraps are copied from jax package.
def curry(f):
  """Curries arguments of f, returning a function on any remaining arguments.

  For example:
  >>> f = lambda x, y, z, w: x * y + z * w
  >>> f(2,3,4,5)
  26
  >>> curry(f)(2)(3, 4, 5)
  26
  >>> curry(f)(2, 3)(4, 5)
  26
  >>> curry(f)(2, 3, 4, 5)()
  26
  """
  return partial(partial, f)

@curry
def wraps(wrapped, fun, namestr="{fun}", docstr="{doc}", **kwargs):
  """
  Like functools.wraps, but with finer-grained control over the name and docstring
  of the resulting function.
  """
  try:
    name = getattr(wrapped, "__name__", "<unnamed function>")
    doc = getattr(wrapped, "__doc__", "") or ""
    fun.__dict__.update(getattr(wrapped, "__dict__", {}))
    fun.__annotations__ = getattr(wrapped, "__annotations__", {})
    fun.__name__ = namestr.format(fun=name)
    fun.__module__ = getattr(wrapped, "__module__", "<unknown module>")
    fun.__doc__ = docstr.format(fun=name, doc=doc, **kwargs)
    fun.__qualname__ = getattr(wrapped, "__qualname__", fun.__name__)
    fun.__wrapped__ = wrapped
  finally:
    return fun

import builtins
import numpy as np
from jax import lax, core
_any = builtins.any
_max = builtins.max
shape = np.shape
from jax._src import dtypes
from jax._src.numpy.lax_numpy import (
    _split_index_for_jit, _merge_static_and_dynamic_indices, _index_to_gather
)

def rewriting_take(arr, idx):
  # Computes arr[idx].
  # All supported cases of indexing can be implemented as an XLA gather,
  # followed by an optional reverse and broadcast_in_dim.

  # Handle some special cases, falling back if error messages might differ.
  if (arr.ndim > 0 and isinstance(idx, (int, np.integer)) and
      not isinstance(idx, (bool, np.bool_)) and isinstance(arr.shape[0], int)):
    if 0 <= idx < arr.shape[0]:
      # Use dynamic rather than static index here to avoid slow repeated execution:
      # See https://github.com/google/jax/issues/12198
      return lax.dynamic_index_in_dim(arr, idx, keepdims=False)
  if (arr.ndim > 0 and isinstance(arr.shape[0], int) and
      isinstance(idx, slice) and
      (type(idx.start) is int or idx.start is None) and
      (type(idx.stop)  is int or idx.stop is  None) and
      (type(idx.step)  is int or idx.step is  None)):
    n = arr.shape[0]
    start = idx.start if idx.start is not None else 0
    stop  = idx.stop  if idx.stop  is not None else n
    step  = idx.step  if idx.step  is not None else 1
    if (0 <= start < n and 0 <= stop <= n and 0 < step and
        (start, stop, step) != (0, n, 1)):
      if _any(isinstance(d, core.Tracer) for d in arr.shape[1:]):
        if step == 1:  # TODO(mattjj, sharadmv): handle step != 1
          return lax.dynamic_slice_in_dim(arr, start, _max(0, stop - start), 0)
      elif step == 1:
        # Use dynamic rather than static slice here to avoid slow repeated execution:
        # See https://github.com/google/jax/issues/12198
        return lax.dynamic_slice_in_dim(arr, start, _max(0, stop - start), 0)
      else:
        return lax.slice_in_dim(arr, start, stop, step)

  # TODO(mattjj,dougalm): expand dynamic shape indexing support
  if jax.config.jax_dynamic_shapes and arr.ndim > 0:
    try: aval = core.get_aval(idx)
    except: pass
    else:
      if (isinstance(aval, core.DShapedArray) and aval.shape == () and
          dtypes.issubdtype(aval.dtype, np.integer) and
          not dtypes.issubdtype(aval.dtype, dtypes.bool_) and
          isinstance(arr.shape[0], int)):
        return lax.dynamic_index_in_dim(arr, idx, keepdims=False)

  treedef, static_idx, dynamic_idx = _split_index_for_jit(idx, arr.shape)
  idx = _merge_static_and_dynamic_indices(treedef, static_idx, dynamic_idx)
  indexer = _index_to_gather(shape(arr), idx)  # shared with _scatter_update
  return indexer

def get_name(funcs: Union[Callable, Sequence]) -> Union[str, Sequence[str]]:
  """Return the name (or namelist) of function 'funcs' (or function list 'funcs').

  Args:
      funcs (Union[Callable, list]): Either a callable, or a list of callable.
  """
  if type(funcs) == list:
    return [w.__name__ for w in funcs]
  return funcs.__name__


def vgd_f (f):
  def _vgd(v, g, l) -> Tuple[jnp.ndarray]:
    """This is a general forward laplacian successor. 
    It takes into a laptuple and the function f to be applied.
    f should satisfy that:
      The output must be a scalar;
      The input should be a pure LapTuple value.
    We use jax.grad and jax.hessian (or jax.linearize) to expand f.

    Returns:
        Tuple[jnp.ndarray]: grad(x), and lap(x)
    """
    ori_shape, n = v.shape, v.size  # vector
    v, g, l = v.reshape((-1,)), g.reshape((-1, n)), l.reshape((-1,))
    _f = lambda x: jnp.squeeze(f(x.reshape(ori_shape)))
    final_value = _f(v)
    # switch to new shape

    # first, calculate grad
    grad_f = jax.grad(_f)
    gf_y, dgrad_f = jax.linearize(grad_f, v)
    final_grad = jnp.matmul(g, gf_y)  # a (len(x), ) vector

    # then, calculate lap
    if lapconfig.autolap.type == lapconfig.autolap.Type.hessian:
      lap_print("    Entering hessian mode.")
      # hessian mode
      final_lap = jnp.sum(gf_y * l) + jnp.sum(
                    jnp.matmul(g, jax.hessian(_f)(v)) * g)
    else:
      # lap_over_f mode.
      lap_print("    Entering fori_loop mode.")
      times = ((n-1) // lapconfig.autolap.block + 1)
      nupper = times * lapconfig.autolap.block
      if n == nupper:
        eye = jnp.eye(n)
        _g = g
      else:
        eye = jnp.concatenate([jnp.eye(n), jnp.zeros((nupper - n, n))], axis = 0)
        _g = jnp.concatenate([g, jnp.zeros((g.shape[0], nupper - n))], axis = 1)
      _g = _g.reshape((g.shape[0], times, -1))
      eye = eye.reshape((times, -1, n))
      v_dgrad_f = jax.vmap(dgrad_f)

      def _loop_he(i, val):
        right = v_dgrad_f(eye[i])
        lap_print(f"    partial hessian shape = {right.shape}")
        right = jnp.matmul(g, right.T)  # (len(x), block)
        lap_print(f"    partial matmul shape = {right.shape}")
        return jnp.sum(_g[:,i] * right) + val
      final_lap = jax.lax.fori_loop(0, times, _loop_he, jnp.sum(gf_y * l))


    return final_value, final_grad, final_lap

  return _vgd

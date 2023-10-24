"""This module includes functional utility tools.
Tools here should not depend on the class in 
  `laptuple.py` and `sparsinfo.py`.
"""
from functools import partial
from typing import Sequence, Union, Callable, Tuple, Callable, TypeVar

import jax
import jax.numpy as jnp
from lapjax.lapsrc.lapconfig import lapconfig

F = TypeVar("F", bound=Callable)


def rewriting_take(arr, idx):
  """
  Computes arr[idx].
  All supported cases of indexing can be implemented as an XLA gather,
  followed by an optional reverse and broadcast_in_dim.
  """
  
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

def get_hash(funcs: Union[Callable, Sequence]) -> Union[int, Sequence[int]]:
  """Return the hash (or hashlist) of function 'funcs' (or function list 'funcs').
  
  Args:
      funcs (Union[Callable, list]): Either a callable, or a list of callable.
  """
  if type(funcs) == list:
    return [w.__hash__() for w in funcs]
  return funcs.__hash__()

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
      lapconfig.log("    Entering hessian mode.")
      # hessian mode
      final_lap = jnp.sum(gf_y * l) + jnp.sum(
                    jnp.matmul(g, jax.hessian(_f)(v)) * g)
    else:
      # lap_over_f mode.
      lapconfig.log("    Entering fori_loop mode.")
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
        lapconfig.log(f"    partial hessian shape = {right.shape}")
        right = jnp.matmul(g, right.T)  # (len(x), block)
        lapconfig.log(f"    partial matmul shape = {right.shape}")
        return jnp.sum(_g[:,i] * right) + val
      final_lap = jax.lax.fori_loop(0, times, _loop_he, jnp.sum(gf_y * l))


    return final_value, final_grad, final_lap

  return _vgd

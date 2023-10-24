from functools import wraps
from typing import Any, Mapping, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from lapjax.lapsrc.lapconfig import lapconfig
from lapjax.lapsrc.laptuple import LapTuple, TupType
from lapjax.lapsrc.laputils import (
  laptupler, lap_counter,
  iter_func, lap_checker, tupler, lap_setter,
)
from lapjax.lapsrc.sparsutils import tuple2spars
from lapjax.lapsrc.function_class import *

def is_wrapped(wrapped_f: F) -> bool:
    return max([wrapped_f.__hash__() in w.hashlist for w in func_type]) == 1

def get_wrap_by_f(wrapped_f: F) -> FBase:
  for wrap_class in func_type:
    if wrapped_f.__hash__() in wrap_class.hashlist:
      return wrap_class
  raise ValueError(f"Function '{wrapped_f.__name__}' is not wrapped.")

def get_wrap_by_type(wrap_type: FType) -> FBase:
  return [w for w in func_type if w.ftype == wrap_type][0]

def custom_wrap(f: F, custom_type: FType, cst_f: F = None, overwrite: bool = False) -> FBase:
  """Wrap a jax function that is not handled by lapjax.
  You can bind `f` to an existing funtion type, 
  or to your customized function `cst_f` if `custom_type=FType.CUSTOMIZED`.
  
  Example: to bind `lapjax.numpy.exp2` that is not handled by lapjax,
  ```
  from lapjax.numpy import exp2
  from lapjax import custom_wrap, FType

  custom_wrap(jnp.exp2, FType.ELEMENT)
  ```

  Args:
    f (F): the function that is not handled by lapjax.
    custom_type (FType): the function type to be binded, could be any FType 
      except FType.OVERLOAD.
    cst_f (F, optional): the customized function with identical input and 
      output structure as `f`, except that the ndarray is replaced to the 
      LapTuple. Defaults to None.
    overwrite (bool, optional): whether to overwrite existing binding.

  FTypes:
    CONSTURCTION: the function is used to construct arrays that does not depend 
      on the input, i.e., the output has zero gradient and laplacian w.r.t. the 
      input. Examples: jnp.ones, jnp.zeros, etc.
    LINEAR: the function is 'linear' in a sense that the out gradient and 
      laplacian could be obtained by directly applied the function. Notice that 
      since part of the lapjax acceleration comes from the sparse 
      representation of the gradient, wrapping a linear function in a general 
      way will cause the sparsity loss. At this point, you could customized the 
      function for LapTuple inputs youself and bind to `CUSTOMIZED` type. 
      Examples: transpose, reshape, etc.
    ELEMENT: the function is element-wise, i.e., each output element is a 
      function of each element of the input. Examples: jnp.sin, jnp.exp, etc.
    MERGING: the function is merging several axes and leaves the rest axes 
      unchanged, or the output is simply a scaler. Examples: jnp.sum, jnp.
      nanmean, etc.
    CUSTOMIZED: functions that do not belong to any FType above. You need to 
      customize a version for LapTuple inputs. Examples: jnp.matmul, jnp.dot, 
      etc.

    Tips:
      1. When warpping new functions, make sure that you understand the 
        sematics correctly. For instance, ReLU can be binded to `ELEMENT` type, 
        but it is not differentiable at the origin. This could cause strange 
        laplacian output.
      2. Bindding a function to `ELEMENT` and `CONSTRUCTION` is welcome, as all 
        acceleration is automatically compatible. However, be careful when 
        bindding to `LINEAR` and `MERGING`. It is possible to cause significant 
        deceleartion.
      3. To bind a function to `CUSTOMIZED`, the major difficulty beyond 
        mathematical derivation is to compute the SparseInfo of the output. You 
        could refer to the existing functions for examples. Generally, you can 
        discard the sparsity across the operating axes first, and then perform 
        the computation normally. An example of `lapjax.numpy.linalg.slogdet` 
        that computes the log value of the determinant of a tensor along the 
        last two axes:
        ```
        def cst_slogdet (*args, **kwargs):  # same as jnp.linalg.slogdet, except that ndarray is replaced to LapTuple
          a: LapTuple = args[0] # LapTuple
          # since it operates on the last two axes, we compute the axis_map as belows
          ax_map = {w:w for w in range(args[0].ndim - 2)}
          ax_map.update({args[0].ndim-2: None, args[0].ndim-1: None})
          a_discard: LapTuple = a.map_axis(ax_map) # discard the sparsity across the last two axes

          # compute the output value, gradient, and laplacian according to mathemtical derivations
          val, grad, lap = a_discard.to_tuple()
          nval = jnp.linalg.slogdet(val)
          ngrad = ...
          nlap = ...
          return nval[0], LapTuple(nval[1], ngrad, nlap, a_discard.spars) # return the sign and the log of determinant.
        ```
        
    """
  assert custom_type in FType, "Custom type should be one of the predefined FTypes."
  if custom_type == FType.OVERLOAD:
    raise ValueError("Overload type is not allowed for custom wrap.")
  if not overwrite and is_wrapped(f):
    raise ValueError(f"Function '{f.__name__}' is already wrapped. " + \
                     "To overwrite, please set overwrite=True.")
  elif is_wrapped(f):
    ori_cls = get_wrap_by_f(f)
    ori_cls.remove_wrap(f)
  wrap_class = get_wrap_by_type(custom_type)
  if custom_type != FType.CUSTOMIZED:
    wrap_class.add_wrap(f)

  else:
    assert cst_f is not None and callable(cst_f), \
      "When custom_type is CUSTOMIZED, cst_f should be a callable function."
    wrap_class.add_wrap(f, cst_f)  
  
  print(f"Successfully bind function '{f.__name__}' to {custom_type}.")
  from lapjax.lapsrc.lapconfig import lapconfig
  if not lapconfig.linear_wrap_warned and custom_type == FType.LINEAR:
    lapconfig.logger.warning(
      "Notice that if custom_type is `FLinear`, " + \
      "the LapTuple might loss the sparsity and cause inefficiency.\n" + \
      "You can customize the function yourself and bind to `CUSTOMIZED`."
    )
    lapconfig.linear_wrap_warned = True
  if not lapconfig.merging_wrap_warned and custom_type == FType.MERGING:
    lapconfig.logger.warning(
      "Notice that if custom_type is `FMerging`, " + \
      "LapJAX uses the traditional hessian-based method " + \
      "to compute the laplacian across the merging axes.\n" + \
      "When there is no sparsity and the merging axes are large, " + \
      "it could cause OOM error.\n" + \
      "You can use the looping mode `lapconfig.set_autolap_fori_loop()`, " + \
      "which might be less efficient, " + \
      "or to customize the function yourself and bind to `CUSTOMIZED`."
    )
    lapconfig.merging_wrap_warned = True
  
  return wrap_class

def lap_dispatcher (wrapped_f: F,
                    i_args: Tuple[Any],
                    i_kwargs: Mapping) -> F:
    """When laptuple is encountered, the dispatcher will behave different based on the wrapper_f.
    When it is a construction function, return operation on VALUE without creating tuples.
    When it is a linear function without changing GRAD range, apply to each LapType directly.
    When it is a linear function with (possibly) merging, check dynamic GRAD sparsity expansion.
    When it is a LapTuple operator, check there are only two inputs (jnp.ndarray or LapTuple).
    When it is a element-wise function, vmap to calculate each LapType, and reshape.
    When it is a customized function, call the custimized function directly.
    Otherwise, raise.

    Args:
        wrapped_f (F): the function that is wrapped.
        i_args (tuple): the boolean tree with identical structure of args. 
          Represent whether elements are laptuple.
        i_kwargs (dict): the bool tree with identical structure of kwargs. 
          Represent whether elements are laptuple.

    Returns:
        F: The lapjax wrapped function that takes tuplized inputs.
    """
    fname = wrapped_f.__name__

    @wraps(wrapped_f)
    def _wrapped_f (*p_args, **p_kwargs):
      """compute the wrapped function on (possibly laptuple) args and kwargs.
      Notice that p_args and p_kwargs are tuplize version, and we convert them back fisrt.
      """
      args, kwargs = laptupler(p_args, i_args), laptupler(p_kwargs, i_kwargs)

      wrap_class = get_wrap_by_f(wrapped_f)
      lapconfig.log(f"==== Dispatch '{fname}' to {wrap_class.classname} class ====")
      wrap_class.examine(wrapped_f, *args, **kwargs)
      return wrap_class.execute(wrapped_f, *args, **kwargs)

    return _wrapped_f

def vmap(fun: F,
        in_axes: Union[int, Sequence[Any]] = 0,
        out_axes: Any = 0,
        ) -> F:
  """
    lapjax wrapped 'vmap' function.
    The expanded vmap supports efficient vectorizing with LapTuple input.
    For original function document, please refer to 'jax.vmap'.    
  """

  from jax.tree_util import tree_flatten, tree_unflatten
  from jax.interpreters import batching
  from jax._src.api_util import flatten_axes

  def _converter (ist, axis):
    """Convert ist to tupler with correct index.
    If axis is None, direct convert.
    Otherwise, swap the grad dimension if axis >= 0.
    """
    if axis is None or axis < 0:
      return ist
    return ist[0], jnp.swapaxes(ist[1], axis, axis+1), ist[2]

  @wraps(fun)
  def _vmap(*args, **kwargs):  # true call from users.
    if lap_counter((args, kwargs)) == 0:
      return jax.vmap(fun, in_axes, out_axes)(*args, **kwargs)

    lapconfig.log("==== Dispatch 'vmap' to customized vmap function ====")
    # process in_axes, and expand them for LapTuple.
    _, in_tree  = tree_flatten((args, kwargs), is_leaf=batching.is_vmappable)
    in_flat = flatten_axes("vmap in_axes", in_tree, (in_axes, 0), kws=True)
    _in = tree_unflatten(in_tree, in_flat)

    # _in shares identical structure with (args, kwargs),
    # with value indicating the axis to be vectorized.
    # For now, we discard the axis before entering _fun.
    def discarder (x: Union[jnp.ndarray, LapTuple], axis: Union[int, None]):
      if axis == None:
        return x
      if not isinstance(x, LapTuple):
        return x
      dic = {w: w if w < axis else (w-1 if w > axis else None) for 
                w in range(x.ndim)}
      return x.map_axis(dic)
    args = laptupler(args, _in[0], discarder)
    kwargs = laptupler(kwargs, _in[1], discarder)
    # Indicator of whether each element is LapTuple.
    # If true, element is `SparsInfo`; None otherwise.
    i_args, i_kwargs = lap_checker(args), lap_checker(kwargs)

    # Create the fully expanded in_axes.
    triple_f = lambda x, y: (x,x if (x is None or x<0) else x+1,x) \
                            if y else x
    tuplized_in = (laptupler(_in[0], i_args, triple_f),
                   laptupler(_in[1], i_kwargs, triple_f))
    lapconfig.log(f"--> Final in_axes: args={tuplized_in[0]}, " +
              f" kwargs={tuplized_in[1]}")

    # The true function to be called.
    def _fun (p_args, p_kwargs):
      out = fun(*laptupler(p_args, i_args),
                **laptupler(p_kwargs, i_kwargs))
      return {"i_out": lap_setter(out, lambda x: isinstance(x, LapTuple)),
              "s_out": lap_setter(out,
                                  lambda x: x.spars.to_tuple() if isinstance(x, LapTuple) else None),
              "p_out": tupler(out)}
    # vmap the function and obtain return vaules (dict).
    v_fun = jax.vmap(_fun,
                     tuplized_in,
                     {"i_out": None, "s_out": None, "p_out": out_axes})
    outs = v_fun(tupler(args), tupler(kwargs))

    # process the out_axes, reduce back to LapTuple.
    out_tree = tree_flatten(outs["p_out"], is_leaf=batching.is_vmappable)[1]
    out_flat = flatten_axes("vmap out_axes", out_tree, out_axes, kws=True)
    # _out is the structurized mapping dimensions
    _out = tree_unflatten(out_tree, out_flat)
    s_out = laptupler(outs["s_out"], outs["i_out"],
                      lambda x, y: tuple2spars(x) if y else False)
    # print(_out, outs["i_out"], s_out)

    # Indicator of whether each element is LapTuple.
    lap_axes = laptupler(_out, outs["i_out"],
                         lambda x, y: x[0] if y else None)
    lapconfig.log("<-- Final out_axes: " + 
              laptupler(_out, outs["i_out"], lambda x, y: x[0] if y else x))

    p_out = laptupler(outs["p_out"], lap_axes, _converter)
    return laptupler(p_out, s_out)

  return _vmap

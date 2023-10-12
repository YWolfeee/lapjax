from functools import wraps
from typing import Any, Mapping, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from lapjax.laptuple import LapTuple, TupType
from lapjax.func_utils import lap_print
from lapjax.laputils import (
  laptupler, lap_counter,
  iter_func, lap_checker, tupler, lap_setter,
  check_single_args, check_pure_kwargs, check_lapcount_args
)
from lapjax.sparsutils import tuple2spars
from lapjax.function_class import *

def is_wrapped(wrapped_f: F):
  return max([max([wrapped_f == u for u in w.funclist]) for w in func_type]) == 1

def lap_dispatcher (wrapped_f: F,
                    i_args: Tuple[Any],
                    i_kwargs: Mapping) -> F:
    """When laptuple is encountered, the dispatcher will behave different based on the wrapper_f.
    When it is a construction function, return operation on VALUE without creating tuples.
    When it is a linear function without changing GRAD range, apply to each LapType directly.
    When it is a linear function with (possibly) merging, check dynamic GRAD sparsity expansion.
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

      if fconstruction.contain(wrapped_f):
        lap_print(f"==== Dispatch '{fname}' to construction function ====")
        return wrapped_f(*iter_func(args), **iter_func(kwargs))

      elif felement.contain(wrapped_f):
        # Take the grad and lap functions, and directly apply.
        # laptuple should not appear in kwargs. In FElement, args are at most one-layer tuple.
        # Only the first element SHOULD BE the laptuple
        lap_print(f"==== Dispatch '{fname}' to element function ====")
        check_pure_kwargs(fname, kwargs)
        check_lapcount_args(fname, args)
        
        return felement._element(wrapped_f, *args, **kwargs)
      
      elif flinear.contain(wrapped_f):
        # Same function applied seperately to VALUE, GRAD, LAP
        # For linear functions, keyword arguments should not be laptuple
        lap_print(f"==== Dispatch '{fname}' to linear function ====")
        check_pure_kwargs(fname, kwargs)
        # Can have multiple LapTuple, e.g. jnp.concatenate
        
        return flinear._linear(wrapped_f, *args, **kwargs)

      elif foverload.contain(wrapped_f):
        # There wouldn't be kwrags, and len(args)==2.
        lap_print(f"==== Dispatch '{fname}' to overload function ====")
        assert len(args) == 2 and len(kwargs) == 0, \
          f"Arguments mismatch. Require binary arguments, but len(args) = {len(args)} and len(kwargs) = {len(kwargs)}."

        return foverload._overload_f(fname, args[0], args[1])

      elif fmerging.contain(wrapped_f):
        lap_print(f"==== Dispatch '{fname}' to merging function ====")
        check_single_args(fname, args)
        check_pure_kwargs(fname, kwargs)

        return fmerging._merging(wrapped_f, *args, **kwargs)

      elif fcustomized.contain(wrapped_f):
        lap_print(f"==== Dispatch '{fname}' to customized function ====")
        return fcustomized._customized(wrapped_f, *args, **kwargs)

      raise NotImplementedError(f"Encounter an unexpected function '{fname}'.")

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

    lap_print("==== Dispatch 'vmap' to customized vmap function ====")
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
    lap_print(f"--> Final in_axes: args={tuplized_in[0]},"
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
    lap_print("<-- Final out_axes:", laptupler(_out,
                                           outs["i_out"],
                         lambda x, y: x[0] if y else x))

    p_out = laptupler(outs["p_out"], lap_axes, _converter)
    return laptupler(p_out, s_out)

  return _vmap

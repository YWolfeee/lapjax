import enum
from functools import partial
from copy import deepcopy
from typing import Tuple

import jax
from jax import vmap
from jax import numpy as jnp
from jax import lax as jlax

from lapjax.func_utils import lap_print, get_name, get_hash, vgd_f, F
from lapjax.axis_utils import get_op_axis
from lapjax.laptuple import LapTuple, TupType
from lapjax.laputils import (
  iter_func, lap_setter, 
  check_single_args, check_pure_kwargs, check_lapcount_args,
)
from lapjax.sparsutils import (
  get_axis_map, 
  shortcut_for_discard, broadcast_spars, 
  matrix_spars, sum_matrix_grad, 
  concat_sparsity,
)


class FType(enum.Enum):
    EMPTY = 0
    CONSTRUCTION = 1
    LINEAR = 2
    ELEMENT = 3
    MERGING = 4
    OVERLOAD = 5
    CUSTOMIZED = 6

class FBase(object):
  support_type = [type(jnp.sum), type(jnp.tanh)]
  funclist = []
  hashlist = []
  classname = "Base"
  ftype = FType.EMPTY

  def __init__(self) -> None:
    for w in self.funclist:  # Can only add function type inside
      assert type(w) in self.support_type, \
      f"{w.__name__} has type {type(w)}, but only support {self.support_type}!"
    self.updated()

  def updated(self) -> None:
    self.namelist = get_name(self.funclist)
    self.hashlist = [w.__hash__() for w in self.funclist]

  def contain(self, f: F):
    return f in self.funclist
  
  def examine(self, f: F, *args, **kwargs):
    """Test whether the function satisfies corresponding properties."""
    pass

  def execute(self, f: F, *args, **kwargs):
    """Execute the function with LapTuple input."""
    pass

  def add_wrap(self, f: F, cst_f: F = None):
    self.funclist.append(f)
    self.updated()

  def __str__(self) -> str:
    return self.classname + ": " + str(self.namelist)


class FLinear(FBase):
  classname = "Linear"
  ftype = FType.LINEAR
  funclist = [
    jnp.reshape, jnp.transpose, jnp.swapaxes,
    jnp.split, jnp.array_split, jnp.concatenate,
    jnp.squeeze, jnp.expand_dims,
    jnp.repeat, jnp.tile,
    jnp.where, jnp.triu, jnp.tril,
    jnp.sum, jnp.mean, #jnp.einsum, # einsum removed.
  ]

  def __init__(self) -> None:
    super(FLinear, self).__init__()

  def examine(self, f: F, *args, **kwargs):
    # Same function applied seperately to VALUE, GRAD, LAP
    # For linear functions, keyword arguments should not be laptuple
    # Can have multiple LapTuple, e.g. jnp.concatenate
    fname = get_name(f)
    check_pure_kwargs(fname, kwargs)

  def execute(self, f, *args, **kwargs):
    fname = get_name(f)
    pf = partial(f, **kwargs)

    # shortcut for mean and sum
    if get_hash(f) in get_hash([jnp.sum, jnp.mean]):
      lap_print(f"|-->`{fname}` tries to operate on dense axes first.")
  
      out =  shortcut_for_discard(f, *args, **kwargs)
      if out is not None: # otherwise, continue the execution
        lap_print(f"|--<`{fname}` shortcut succeeds.")
        return out
      lap_print(f"|--<`{fname}` shortcut fails, will behave normally.")

    # Ensure that the call is normal.
    # Standard bug will be raised here.
    val_out = pf(*iter_func(args))    

    if get_hash(f) == get_hash(jnp.concatenate):
      check_single_args(fname, args)
      op_axis = kwargs.get("axis", 0)
      
      if isinstance(args[0], LapTuple): # regard the first dim as list
        op_axis = (op_axis + int(op_axis >= 0)) % args[0].ndim
        l_args = (args[0].discard({op_axis}), )
        spars = l_args[0].spars
      elif len(args[0]) == 1: # Fake concatenate
        return args[0][0]
      else: # args[0] should be a list / tuple
        # check sparsities, and obtain the resulting spars
        # grads are properly discarded upon return.
        arrays, spars = concat_sparsity(args[0], op_axis)
        l_args = (arrays, )
      lap_print(f"    Discard sparsity to {spars.tups}.")

    else: 
      array: LapTuple = args[0]
      ax_map = get_axis_map(f, *args, **kwargs)
      if get_hash(f) in get_hash([jnp.split, jnp.array_split]):
        split_axis = [k for k,v in ax_map.items() if v is None][0]
        if array.spars.check_leading_axis(split_axis):
          # array remains unchanged.
          l_spars = array.spars.split_sparsity(axis=split_axis, 
                                               size=[w.shape[split_axis] for w in val_out])
        else:
          array = array.map_axis(ax_map)
          l_spars = [deepcopy(array.spars) for _ in val_out]
      else:
        array = array.map_axis(ax_map)
        spars = array.spars

      l_args = (array, ) + args[1:]

    lap_out = pf(*iter_func(l_args, opargs=(TupType.LAP, )))

    in_axes = lap_setter(l_args,
                         lambda x: 0 if isinstance(x, LapTuple) else None)
    vpf = vmap(pf, in_axes)
    grad_out = vpf(*iter_func(l_args, opargs=(TupType.GRAD,)))

    # Notice that, output is either an array, or an array list
    if isinstance(val_out, jnp.ndarray):
      return LapTuple(val_out, grad_out, lap_out, spars)
    else:  # must be a list
      return [LapTuple(v,g,l,s) for v,g,l,s in zip(
                      val_out, grad_out, lap_out, l_spars)]


class FConstruction(FBase):
  classname = "Construction"
  ftype = FType.CONSTRUCTION
  funclist = [
    jnp.shape, jnp.size,
    jnp.eye, jnp.array,
    jnp.ones, jnp.ones_like,
    jnp.zeros, jnp.zeros_like,
    jnp.asarray, jnp.sign,
    jlax.stop_gradient,
  ]

  def __init__(self) -> None:
    super(FConstruction,self).__init__()

  def execute(self, f: F, *args, **kwargs):
    return f(*iter_func(args), **iter_func(kwargs))


class FElement(FBase):
  classname = "Element-wise"
  ftype = FType.ELEMENT
  funclist = [
    jnp.sin, jnp.cos, jnp.tan,
    jnp.arcsin, jnp.arccos, jnp.arctan,
    jnp.arcsinh, jnp.arccosh, jnp.arctanh,
    jnp.sinh, jnp.cosh, jnp.tanh,
    jnp.exp, jnp.log,
    jnp.square, jnp.sqrt, jnp.power,
    jnp.abs, jnp.absolute,
    jlax.sin, jlax.cos, jlax.tan,
    jlax.asin, jlax.acos, jlax.atan, jlax.atan2,
    jlax.asinh, jlax.acosh, jlax.atanh, 
    jlax.exp, jlax.log, 
    jlax.square, jlax.sqrt, jlax.rsqrt, 
    jlax.pow, jlax.integer_pow,
    jlax.abs, 
  ]

  def __init__(self) -> None:
    super(FElement,self).__init__()

  def examine(self, f: F, *args, **kwargs):
    # laptuple should not appear in kwargs. In FElement, args are at most one-layer tuple.
    # Only the first element SHOULD BE the laptuple
    fname = get_name(f)
    check_pure_kwargs(fname, kwargs)
    check_lapcount_args(fname, args)

  def execute(self, _f, *args, **kwargs):
    def p_f (x):
      cargs = (x,)+args[1:]
      return _f(*cargs, **kwargs)
    flist = [p_f, jax.grad(p_f), jax.grad(jax.grad(p_f))]

    x, g, l = args[0].to_tuple()
    out = [vmap(f)(x.reshape(-1)).reshape(x.shape) for f in flist]

    return LapTuple(out[0], out[1][None] * g,
            out[1] * l + out[2] * jax.numpy.sum(g ** 2, axis=0),
            spars=args[0].spars)


class FOverload(FBase):
  classname = "Overload"
  ftype = FType.OVERLOAD
  funclist = [
    jnp.add, jnp.subtract, jnp.multiply, jnp.divide, jnp.true_divide
  ]

  def __init__(self) -> None:
    super(FOverload, self).__init__()

  def examine(self, f: F, *args, **kwargs):
    # There wouldn't be kwrags, and len(args)==2.
    fname = get_name(f)
    assert len(args) == 2 and len(kwargs) == 0, \
      f"Arguments mismatch. Require binary arguments, but len(args) = {len(args)} and len(kwargs) = {len(kwargs)}."

  def execute (self, fname, *args, **kwargs):
    del kwargs
    x1, x2 = args
    if fname == "add":
      return x1 + x2
    elif fname == "subtract":
      return x1 - x2
    elif fname == "multiple":
      return x1 * x2
    elif fname in ["divide", "true_divide"]:
      return x1 / x2


class FMerging(FBase):
  classname = "Merging"
  ftype = FType.MERGING
  funclist = [
    jnp.linalg.norm, jnp.prod,
  ]

  def __init__(self) -> None:
    super(FMerging, self).__init__()

  def examine(self, f: F, *args, **kwargs):
    fname = get_name(f)
    check_single_args(fname, args)
    check_pure_kwargs(fname, kwargs)

  def execute (self, f, *args, **kwargs):
    fname = get_name(f)
    """
    These functions always operate along some axes.
    We vmap the rest axises, such that the function outputs a scalar,
    and use jax built-in function to compute the output LapTuple.

    As we directly calculate the hessian to obtain the laplacian,
    functions here might result in an OOM issue,
    especially when they involve operations on a huge matrix.
    """
    check_single_args(fname, args)
    _args, _kwargs = iter_func(args), iter_func(kwargs)


    # obtain the operating axises, and remove the `axis` argument.
    op_axis, p_args, p_kwargs = get_op_axis(*_args, **_kwargs)

    # short-cut
    if get_hash(f) == get_hash(jnp.linalg.norm):
      from lapjax import numpy as my_jnp
      if p_kwargs.get("ord") in [None, 'fro'] or \
        (p_kwargs.get("ord") == 2 and len(op_axis) == 1):
        lap_print("----Calling sqrt-sum-square sequence as shortcut of 'norm'.")
        new_args = (my_jnp.square(args[0]), ) + args[1:]
        kwargs.pop('ord', None)
        return my_jnp.sqrt(my_jnp.sum(*new_args, **kwargs))

    # Since returned op_axis is non-negative, this returns vmap axis.
    array: LapTuple = args[0]
    array = array.discard(op_axis)

    length = len(array.get(TupType.VALUE).shape)
    vmap_axis = list(set(range(length)) - op_axis)

    # Construct the pure function.
    # Notice that axis info is removed, so return value is a scalar.
    v_f = vgd_f(partial(f, *p_args[1:], **p_kwargs))

    for axis in vmap_axis:
      v_f = vmap(v_f, in_axes=(axis, axis + 1, axis), out_axes=(-1, -1, -1))

    value = f(*_args, **_kwargs)
    _, grad, lap = v_f(*array.to_tuple())
    # consider keep_dims
    grad = grad.reshape((grad.shape[0],)+ value.shape)
    lap = lap.reshape(value.shape)

    return LapTuple(value, grad, lap, array.spars)


class FCustomized(FBase):
  classname = "Customized"
  ftype = FType.CUSTOMIZED
  funclist = [
    jnp.matmul, jnp.dot,
    jnp.max, jnp.min,
    jnp.amax, jnp.amin,
    jnp.linalg.slogdet,
    jax.nn.logsumexp,
    jax.nn.softmax
  ]

  custom_dict = {}  # user defined function

  def __init__(self) -> None:
    super(FCustomized, self).__init__()

  def add_wrap(self, f: F, cst_f: F):
    super().add_wrap(f, cst_f)
    self.custom_dict[f.__hash__()] = cst_f  # call cst_f when f is called

  def execute(self, f, *args, **kwargs):
    fname = get_name(f)
    if get_hash(f) in get_hash([jnp.max, jnp.min, jnp.amax, jnp.amin,]):
      # Consider shortcut first:
      lap_print(f"|-->`{fname}` tries to operate on dense axes first.")
      out =  shortcut_for_discard(f, *args, **kwargs)
      if out is not None: # otherwise, continue the execution
        lap_print(f"|--<`{fname}` shortcut succeeds.")
        return out
      lap_print(f"|--<`{fname}` shortcut fails, will behave normally.")

    # Behave according to the function.
    if get_hash(f) in get_hash([jnp.matmul, jnp.dot]):
      _f, r_args = partial(f, **kwargs), args
      _f.__hash__ = f.__hash__ # ensure the hash is the same

      try:
        x1, x2 = r_args
      except:
        raise IndexError(f"{fname} requires 2 array arguments, but got {len(r_args)}.")
      # f takes r_args as inputs, where each argument is a ndarray or laptuple

      if x1.ndim == 0 or x2.ndim == 0 and get_hash(f) == get_hash(jnp.dot):
        # handle valid scalar inputs
        return x1 * x2

      value = _f(*iter_func(r_args))
      if isinstance(x1, LapTuple) and isinstance(x2, LapTuple):
        assert x1.spars.get_id() == x2.spars.get_id()
        v1, g1, l1 = x1.to_tuple()
        v2, g2, l2 = x2.to_tuple()
        
        # grad = vf1(g1, v2) + vf2(v1, g2)
        s1, vg1 = matrix_spars(0, x1.spars, _f, g1, v2)
        s2, vg2 = matrix_spars(1, x2.spars, _f, v1, g2)
        spars, lg = broadcast_spars([s1, s2], [vg1, vg2])
        grad = lg[0] + lg[1]

        cross_lap = sum_matrix_grad(_f, [x1.spars, x2.spars], [g1, g2])
        if cross_lap is None:
          g1 = x1.spars.set_dense(g1, True)
          g2 = x2.spars.set_dense(g2, True)
          cross_lap = jnp.sum(vmap(_f)(g1, g2), axis=0)
        lap = _f(v1, l2) + _f(l1, v2) + 2 * cross_lap
      else:  # only one of them is laptup
        lap = _f(*iter_func(r_args, opargs=(TupType.LAP,)))

        if isinstance(x1, LapTuple):
          spars, lap_idx, g1, g2 = x1.spars, 0, x1.grad, x2
        elif isinstance(x2, LapTuple):
          spars, lap_idx, g1, g2 = x2.spars, 1, x1, x2.grad
        spars, grad = matrix_spars (lap_idx, spars, _f, g1, g2)
        
      return LapTuple(value, grad, lap, spars)

    elif get_hash(f) in get_hash([jnp.max, jnp.amax, jnp.min, jnp.amin]):
      # To accurately return the GRAD and LAP arrays, we use mask.
      # Keep array dim and mask the maximum, and then index by the mask.
      # Notice that keepdims argument should not be pass to args.
      check_pure_kwargs(fname, kwargs)
      check_single_args(fname, args)
      p_args, p_kwargs = iter_func(args), iter_func(kwargs)
      p_kwargs['keepdims'] = True

      # get LapTuple and discard grad if required.
      array: LapTuple = args[0].map_axis(
          get_axis_map(f, *args, **kwargs))

      # can have multiple True
      mask = array.get(TupType.VALUE) == f(*p_args, **p_kwargs)

      kwargs.update({"where": mask})
      def index_arr (arr):
        r_args = (arr,) + args[1:]
        return jnp.mean(*r_args, **kwargs)
      v_index_arr = vmap(index_arr, 0, 0)

      return LapTuple(index_arr(array.get(TupType.VALUE)),
                      v_index_arr(array.get(TupType.GRAD)),
                      index_arr(array.get(TupType.LAP)),
                      array.spars
                      )
    elif get_hash(f) == get_hash(jnp.linalg.slogdet):
      det = f(*iter_func(args), **iter_func(kwargs))

      # get LapTuple and discard grad if required.
      array: LapTuple = args[0].map_axis(
          get_axis_map(f, *args, **kwargs))

      valx, gradx, lapx = array.to_tuple()
      inv = jnp.linalg.inv(valx)

      xinv_nabla = jnp.matmul(inv, gradx)

      # compute the gradient
      r_grad = jnp.trace(xinv_nabla, axis1 = -1, axis2 = -2)

      # compute the laplaican
      lap = jnp.trace(jnp.matmul(inv, lapx), axis1 = -1, axis2 = -2)
      matlen = len(xinv_nabla.shape)
      trans = tuple(range(matlen-2)) + (matlen - 1, matlen - 2)
      lap -= jnp.sum(xinv_nabla * jnp.transpose(xinv_nabla, trans),
                     axis=(-2,-1,0))

      return det[0], LapTuple(det[1], r_grad, lap, array.spars)

    elif get_hash(f) == get_hash(jax.nn.logsumexp):
      check_pure_kwargs(fname, kwargs)
      check_single_args(fname, args)

      # get LapTuple and discard grad if required.
      array: LapTuple = args[0].map_axis(
          get_axis_map(f, *args, **kwargs))

      v, g, l = array.to_tuple()
      # ensure non-negative
      # axis = mod_axis(kwargs.get("axis", None), v.ndim)
      axis= tuple(get_op_axis(*iter_func(args), **iter_func(kwargs))[0])
      b = kwargs.get("b", jnp.ones_like(v))
      return_sign = kwargs.get("return_sign", False)
      keepdims = kwargs.get("keepdims", False)

      # keepdims first
      kwargs["keepdims"] = True
      outs = f(*iter_func(args), **kwargs)
      value, sign = outs if return_sign else (outs, jnp.ones_like(outs))

      nabla = jnp.exp(v - value) * b * sign
      nablamul = g * nabla[None]
      grad = jnp.sum(nablamul,
                     lap_setter(axis, lambda x: None if x is None else x+1),
                     keepdims=keepdims)

      lap = jnp.sum(jnp.sum(nablamul * g, axis = 0) + nabla*l,
                     axis = axis,
                     keepdims=keepdims)
      lap -= jnp.sum(grad ** 2, axis = 0)

      if not keepdims:
        value = jnp.squeeze(value, axis=axis)
        sign = jnp.squeeze(sign, axis=axis)
      if return_sign:
        return LapTuple(value, grad, lap, array.spars), sign
      else:
        return LapTuple(value, grad, lap, array.spars)

    elif get_hash(f) == get_hash(jax.nn.softmax):
      check_single_args(fname, args)
      check_pure_kwargs(fname, kwargs)
      kwargs['keepdims'] = True
      kwargs["axis"] = kwargs.get("axis", (-1,))
      from lapjax import numpy as my_jnp

      # get LapTuple and discard grad if required.
      array: LapTuple = args[0].map_axis(
          get_axis_map(f, *args, **kwargs))
      p_args = (array,) + args[1:]
      x_max = my_jnp.max(*p_args, **kwargs)

      from lapjax.lax import stop_gradient
      # TODO: postpone the discard of array when calling substract.
      unnormalized = my_jnp.exp(array - stop_gradient(x_max))
      r_args = (unnormalized,) + args[1:]
      return unnormalized / my_jnp.sum(*r_args, **kwargs)
    
    elif get_hash(f) in self.custom_dict.keys():  # user defined function
      return self.custom_dict[get_hash(f)](*args, **kwargs)

fconstruction = FConstruction()
flinear = FLinear()
felement = FElement()
foverload = FOverload()
fmerging = FMerging()
fcustomized = FCustomized()
func_type: Tuple[FBase] = (fconstruction, flinear, felement, foverload, fmerging, fcustomized)

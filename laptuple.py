from typing import Any, Tuple, Union, Set
from copy import deepcopy
import enum

import jax
import jax.numpy as jnp

from lapjax.axis_utils import AX_MAP
from lapjax.func_utils import lap_print, rewriting_take
from lapjax.sparsinfo import SparsInfo, InputInfo

class TupType(enum.Enum):
  VALUE = 0
  GRAD = 1
  LAP = 2

class LapTuple(object):
  """Forward Laplacian wrapped ndarray, with its nabla and laplacian w.r.t input.
  We support normal operations on jax.numpy.ndarray for LapTuple, where value
  is processed indentically, and grad and laplacian is calculated accordingly.
  
  """
  def __init__(self, value:jnp.ndarray,
                     grad:jnp.ndarray = None,
                     lap: jnp.ndarray = None,
                     spars: SparsInfo = None,
                     *,
                     is_input: bool = False):
    """Initialize LapTuple. 
    value is necessary.
    (1) Direct Construct (Require sparsity and all arrays)
    When grad is not None, lap and spars cannot be None, while 
      is_input must be False.
      In such case, value.shape = grad.shape[1:] = lap.shape;
      We explicitly check the dimension of grad.
    (2) Create constant (No gradient, require sparsity)
    When grad is None, lap must be None. 
      sparsity tells the shape of the zero gradient, i.e.
      ```
      grad = jnp.zeros((spars.get_gdim(),) + value.shape),
      lap  = jnp.zeros_lie(value)
      ```
    
    (3) Create for input (No sparsity, grad, and lap)
    When is_input is True, grad, lap, and spars should be None.
      In such case, value is regraded as the initial input, i.e.
      grad = jnp.eye(value.shape[0]), lap = jnp.zeros_like(value)
      New InputInfo instance will be created accordingly.

    Args:
        value (jnp.ndarray): The ndarray to be converted.
        grad (jnp.ndarray, optional): Its gradient to input. Defaults to None.
          We have grad.shape == input.shape + value.shape. 
        lap (jnp.ndarray, optional): Its laplacian to input. Defaults to None.
          We have lap.shape == value.shape
        spars (SparsTuple, optional): The sparsity tuple of gradient.
        is_input (bool, optional): Whether value is the initial Input. 
          Defaults to False.
    """
    self.value = value
    self.shape = self.value.shape
    self.size = self.value.size
    self.ndim = len(self.shape)

    if is_input:
      assert grad is None and lap is None, \
        "When `is_input` == True, only `value` should be passed."
      assert self.ndim >= 1, "When `is_input` == True, value must be an array."
      # self.grad = jnp.eye(value.shape[0], dtype=value.dtype)
      self.grad = jnp.ones_like(value)[None]
      self.lap = jnp.zeros_like(value, dtype=value.dtype)

      # Creating Sparsity
      info = InputInfo(self.size)
      self.spars = SparsInfo(info, is_input=True, ndim=self.ndim)
      return

    if grad is not None:  # pass all ndarray arguments.
      assert grad is not None and lap is not None and spars is not None, \
        "When `n_input` is None and `is_input` == False, " + \
        "`grad`, `lap` and `spars` should be passed."
      assert grad.shape[1:] == value.shape and lap.shape == value.shape, \
        "Shapes mismatch when constructing LapTuple. " + \
        f"Value: {value.shape}, grad: {grad.shape}, lap: {lap.shape}"
      assert grad.shape[0] == spars.get_gdim(), \
        (f"`grad` and `spars` mismatch. Gradient dim is {grad.shape[0]}, "
        f"while `spars` has {spars.get_gdim()} gradient dim.")
      self.grad = grad
      self.lap = lap
      self.spars = spars
    else:
      assert grad is None and lap is None and spars is not None, \
        ("When constructing non-input LapTuple with None `grad`,"
        " `lap` should be None and `spars` should be passed.")
      self.grad = jnp.zeros((spars.get_gdim(), ) + value.shape, 
                            dtype=value.dtype)
      self.lap = jnp.zeros_like(value, dtype=value.dtype)
      self.spars = spars

  def __repr__(self) -> str:
    return str(self)

  def discard (self, op_axis: Set[int]) -> "LapTuple":
    os = self.grad.shape
    new_spars = deepcopy(self.spars)
    grad = new_spars.discard(op_axis, self.grad)
    # Upon return, self.spars has already been changed.
    if os != grad.shape:
      lap_print(f"  Discard grad shape from {os} to {grad.shape}")
    return LapTuple(self.value, grad, self.lap, new_spars)
  
  def set_dense(self, force = False) -> "LapTuple":
    new_spars = deepcopy(self.spars)
    grad = new_spars.set_dense(self.grad, force=force)
    return LapTuple(self.value, grad, self.lap, new_spars)

  def set_irange(self, irange: Tuple[int]) -> "LapTuple":
    new_spars = deepcopy(self.spars)
    grad = new_spars.set_irange(irange, self.grad)
    return LapTuple(self.value, grad, self.lap, new_spars)

  def map_axis(self, ax_map: AX_MAP) -> "LapTuple":
    """Convert the SparsInfo based on the axis mapping dict.
    ax_map should contain all keys in range(self.ndim).
    The value can be a new axis index, or None.
    If not None, we map the original index to the new one;
    If None, this axis should be discarded.

    """
    if 0 not in ax_map.keys():  # keys are tuple
      return self.map_axis_tuple(ax_map)
    os = self.grad.shape
    new_spars = deepcopy(self.spars)
    grad = new_spars.discard(set([w for w,v in ax_map.items() if v is None]), 
                             self.grad)
    new_spars.swap_axis({w:v for w,v in ax_map.items() if v is not None})
    if os != grad.shape:
      lap_print(f"  Update grad shape from {os} to {grad.shape}")
    return LapTuple(self.value, grad, self.lap, new_spars)

  def map_axis_tuple (self, ax_map: AX_MAP) -> "LapTuple":
    """Same as map_axis, but the key and values are int tuple.
    This is for jnp.reshape with a more complicated scenario."""
    os = self.grad.shape
    new_spars = deepcopy(self.spars)
    grad = new_spars.map_for_reshape(self.grad, ax_map)
    if os != grad.shape:
      lap_print(f"  Update grad shape from {os} to {grad.shape}")
    return LapTuple(self.value, grad, self.lap, new_spars)

  def get(self, key: TupType) -> jnp.ndarray:
    return self.to_tuple()[key.value]

  def reshape(self, *args: Any, order: str = "C") -> 'LapTuple':
    from lapjax import numpy as my_jnp
    if type(args[0]) != tuple:  # encap the axis dim
      args = (args,)
    return my_jnp.reshape(self, *args, order=order)

  def transpose(self, *args: Any) -> 'LapTuple':
    from lapjax import numpy as my_jnp
    if not args:
      return my_jnp.transpose(self)
    if type(args[0]) != tuple:  # encap the axis dim
      args = (args,)
    return my_jnp.transpose(self, *args)

  def __iter__ (self) -> "LapTuple":
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")  # same as numpy error
    self.iterable = 0
    return self

  def __next__ (self):
    if self.iterable < self.value.shape[0]:
      self.iterable += 1
      return self[self.iterable-1]
    raise StopIteration

  def __getitem__(self, key) -> 'LapTuple':
    vs = jax.vmap(lambda x, k: x[k], in_axes=(0,None), out_axes=(0))
    if type(key) != tuple:
      key = (key,)
    indexer = rewriting_take(self.value, idx=key)  # jax indexer API

    # discard gradient and map spars accordingly
    _self = self.map_axis(sutils.getitem_map(indexer, self.ndim))
    _self = LapTuple(_self.value[key], vs(_self.grad, key), _self.lap[key], _self.spars)

    # handle `None` in key
    map_to = [w for w in range(_self.value.ndim) if \
                w not in indexer.newaxis_dims]
    _self.spars.swap_axis({i: w for i, w in enumerate(map_to)})
    
    return _self

  def to_tuple(self) -> Tuple[jnp.ndarray]:
    return self.value, self.grad, self.lap

  def __neg__(self) -> 'LapTuple':
    return LapTuple(-self.value, -self.grad, -self.lap, self.spars)

  def __add__(self, x: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    if isinstance(x, LapTuple):
      v = self.value + x.value
      assert self.spars.get_id() == x.spars.get_id()
      spars, gs = sutils.broadcast_spars([self.spars, x.spars], 
                                         [self.grad, x.grad])
      g = gs[0] + gs[1]
      l = self.lap + x.lap
      return LapTuple(v,g,l,spars)

    if isinstance(x, float) or isinstance(x, int):
      return LapTuple(self.value + x, self.grad, self.lap, self.spars)

    if isinstance(x, jnp.ndarray):
      vshape = jnp.broadcast_shapes(self.value.shape, x.shape)
      spars = deepcopy(self.spars)
      grad = spars.broadcast_dim(vshape, self.grad)
      return LapTuple(value=self.value + x, 
                      grad=grad, 
                      lap=jnp.broadcast_to(self.lap, vshape), 
                      spars=spars)

  def __radd__(self, x: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    return self + x

  def __sub__(self, x: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    return self.__add__(-x)

  def __rsub__(self, x: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    return (-self) + x

  def __mul__(self, x: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    if isinstance(x, LapTuple):
      return lap_mul(self,x)
    if isinstance(x, float) or isinstance(x,int):
      return LapTuple(self.value*x,self.grad*x,self.lap*x,self.spars)
    if isinstance(x,jnp.ndarray):
      vshape = jnp.broadcast_shapes(self.value.shape, x.shape)
      spars = deepcopy(self.spars)
      grad = spars.broadcast_dim(vshape, self.grad)
      vm = jax.vmap(jnp.multiply, (0, None), 0)
      return LapTuple(self.value*x, vm(grad,x), self.lap*x, spars)

  def __rmul__(self, x: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    return self*x

  def __truediv__(self, x: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    if isinstance(x, LapTuple):
      return lap_mul(self, reciprocal_withL(x))
    if isinstance(x, float) or isinstance(x,int):
      return LapTuple(self.value/x,self.grad/x,self.lap/x, self.spars)
    if isinstance(x,jnp.ndarray):
      vshape = jnp.broadcast_shapes(self.value.shape, x.shape)
      spars = deepcopy(self.spars)
      grad = spars.broadcast_dim(vshape, self.grad)
      vd = jax.vmap(jnp.divide, (0,None))
      return LapTuple(self.value/x, vd(grad,x), self.lap/x, spars)

  def __rtruediv__(self, x: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    # x could't be LapTuple, or __truediv__ will be called.
    return reciprocal_withL(self) * x

  def __pow__(self, p: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    from lapjax import numpy as my_jnp
    return my_jnp.power(LapTuple(self.value, self.grad, self.lap, self.spars), p)

  def __str__(self) -> str:
    return 'LapTuple(\n  Value: \n{}\n  Grad: \n{} \n  Laplacian: \n{}\n)'.format(self.value, self.grad, self.lap)


def reciprocal_withL(input_x: LapTuple) -> LapTuple:
  x, x_nabla, x_nabla2 = input_x.to_tuple()
  y = 1/x
  y_nabla = -x_nabla/(x[None]**2)
  y_nabla2 = -x_nabla2/(x**2) + \
             2 * jnp.sum(x_nabla ** 2,axis=0) * (y**3)
  return LapTuple(y, y_nabla, y_nabla2, input_x.spars)

def lap_mul (x: LapTuple, y: LapTuple) -> LapTuple:
  v = x.value * y.value
  assert x.spars.get_id() == y.spars.get_id()
  spars, gs = sutils.broadcast_spars([x.spars, y.spars], 
                                     [x.grad, y.grad])
  g = gs[0] * y.value[None] + gs[1] * x.value[None]
  l = 2 * jnp.sum(gs[0] * gs[1], axis=0) + y.lap * x.value + x.lap * y.value
  return LapTuple(v, g, l, spars)

import lapjax.sparsutils as sutils

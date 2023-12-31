from typing import Any, Tuple, Union, Set
from copy import deepcopy
import enum

import jax
import jax.numpy as jnp

from lapjax.lapsrc.lapconfig import lapconfig
from lapjax.lapsrc.axis_utils import AX_MAP
from lapjax.lapsrc.func_utils import rewriting_take
from lapjax.lapsrc.sparsinfo import SparsInfo, InputInfo

class TupType(enum.Enum):
  VALUE = 0
  GRAD = 1
  LAP = 2

class LapTuple(object):
  def __init__(self, value:jnp.ndarray,
                     grad:jnp.ndarray = None,
                     lap: jnp.ndarray = None,
                     spars: SparsInfo = None,
                     *,
                     is_input: bool = False):
    """
    `LapTuple` is a ternary array group (an array with its nabla and laplacian w.r.t the input specified in its `SparsInfo`). This is the basic data structure for `LapJAX`.
    We support various common jax operations for LapTuple, where value
    is processed indentically, and grad and laplacian is calculated according to 
    mathematical rules. For example, for LapTuple x and y, we have:
    ```
    x + y = LapTuple(x.value + y.value, 
                     x.grad + y.grad, 
                     x.lap + y.lap)
  
    x * y = LapTuple(x.value * y.value, 
                     x.grad * y.value[None] + y.grad * x.value[None],
                     2 * jnp.sum(x.grad * y.grad, axis=0) + x.lap * y.value + y.lap * x.value)
    ``` 

    ### Data Structure
    Specifically, we have:
    ```
      value.shape == grad.shape[1:] == lap.shape
    ```
    Here, `grad.shape[0]` is the gradient dimension, which is determined by the sparsity. When there is no sparsity, `grad.shape[0]` equals the size of the 
    input, and `grad[idx]` is the gradient of `value` w.r.t the `input[idx]`. In
    most cases, `grad.shape[0]` is much smaller than the size of the input due to
    the sparsity. See `SparsInfo` for how we handle the sparsity.
    To obtain the value, gradient, and laplacian, use `.get(TupType)` method, or
    simply use `LapTuple.value, LapTuple.grad, LapTuple.lap`.
  
    The sparsity contributes to the major acceleration of LapJAX. See `SparsInfo` for more details.
    
    ### Constructing LapTuple
    To construct a LapTuple for the input array `x`, simply use 
    ```
    from lapjax import LapTuple, numpy
    x = numpy.eye(3)
    LapTuple(x, is_input=True)
    x.shape # (3, 3)
    ```
    It is highly unrecommended to manually construct a LapTuple for non-input array, unless you know exactly what you want. This is likely to break the sparsity and harm the efficiency severely. To construct a LapTuple w.r.t input x with grad zero, use
    ```
    from lapjax import LapTuple, SparsInfo, numpy
    x = LapTuple(numpy.eye(3), is_input=True)
    y = LapTuple(numpy.eye(4), spars=SparsInfo(x.spars.input))
    y.grad.shape  # (9, 4, 4)
    ```
    To construct a LapTuple w.r.t input x manually, use
    ```
    from lapjax import LapTuple, SparsInfo
    y = LapTuple(value, grad, lap, spars=SparsInfo(x.spars.input)
    ```
    Again, specify the input and let LapTuple handle the sparsity when possible.

    ### Parameters
    value (jnp.ndarray): The ndarray to be converted.
    grad (jnp.ndarray, optional): The gradient of value w.r.t. input. Defaults to 
      None. We have `grad.shape == (d,) + value.shape`, where `d<=value.size` is 
      the sparse gradient dimension. 
    lap (jnp.ndarray, optional): The laplacian of value w.r.t. input. Defaults to 
      None. We have `lap.shape == value.shape`.
    spars (SparsTuple, optional): The sparsity tuple of gradient. See `SparsInfo`.
    is_input (bool, optional): Whether this is the input. Defaults to False.
    """
    self.value = deepcopy(value)
    self.shape = self.value.shape
    self.size = self.value.size
    self.ndim = len(self.shape)
    self.identical = self.__identical
    self.dtype = self.value.dtype

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
      self.grad = deepcopy(grad)
      self.lap = deepcopy(lap)
      self.spars = deepcopy(spars)
    else:
      assert grad is None and lap is None and spars is not None, \
        ("To construct LapTuple from input, set `is_input` to `True`.\n"
         "To construct non-input LapTuple with zero `grad`,"
         " `lap` should be None and `spars` should be passed.")
      self.grad = jnp.zeros((spars.get_gdim(), ) + value.shape,
                            dtype=value.dtype)
      self.lap = jnp.zeros_like(value, dtype=value.dtype)
      self.spars = deepcopy(spars)

  def __repr__(self) -> str:
    return 'LapTuple(\n  Value:\n{}\n  Grad:\n{}\n  Laplacian:\n{}\n)'.format(self.value.__repr__(), self.grad.__repr__(), self.lap.__repr__())

  def discard(self, op_axis: Set[int]) -> "LapTuple":
    os = self.grad.shape
    new_spars = deepcopy(self.spars)
    grad = new_spars.discard(op_axis, self.grad)
    # Upon return, self.spars has already been changed.
    if os != grad.shape:
      lapconfig.log(f"  Discard grad shape from {os} to {grad.shape}")
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

    If ax_map is {}, we discard all axis, lossing sparsity.

    """
    if ax_map == {}:
      return self.set_dense(force=True)
    if 0 not in ax_map.keys():  # keys are tuple
      return self.map_axis_tuple(ax_map)
    os = self.grad.shape
    new_spars = deepcopy(self.spars)
    grad = new_spars.discard(set([w for w, v in ax_map.items() if v is None]),
                             self.grad)
    new_spars.swap_axis({w: v for w, v in ax_map.items() if v is not None})
    if os != grad.shape:
      lapconfig.log(f"  Update grad shape from {os} to {grad.shape}")
    return LapTuple(self.value, grad, self.lap, new_spars)

  def map_axis_tuple(self, ax_map: AX_MAP) -> "LapTuple":
    """Same as map_axis, but the key and values are int tuple.
    This is for jnp.reshape with a more complicated scenario."""
    os = self.grad.shape
    new_spars = deepcopy(self.spars)
    grad = new_spars.map_for_reshape(self.grad, ax_map)
    if os != grad.shape:
      lapconfig.log(f"  Update grad shape from {os} to {grad.shape}")
    return LapTuple(self.value, grad, self.lap, new_spars)

  def get(self, key: TupType) -> jnp.ndarray:
    return self.to_tuple()[key.value]

  def reshape(self, *args: Any, order: str = "C") -> 'LapTuple':
    from lapjax import numpy as my_jnp
    if type(args[0]) != tuple:  # encap the axis dim
      args = (args,)
    return my_jnp.reshape(self, *args, order=order)

  def transpose(self, *args, **kwargs) -> 'LapTuple':
    from lapjax import numpy as my_jnp
    if len(args) > 0 and type(args[0]) != tuple:
      args = (args,)
    axes = (kwargs['axes'],) if 'axes' in kwargs else args
    return my_jnp.transpose(self, *axes)

  @property
  def T(self) -> 'LapTuple':
    return self.transpose()

  def __iter__(self) -> "LapTuple":
    if self.ndim == 0:
      # same as numpy error
      raise TypeError("iteration over a 0-d array")
    self.iterable = 0
    return self

  def __next__(self):
    if self.iterable < self.value.shape[0]:
      self.iterable += 1
      return self[self.iterable-1]
    raise StopIteration

  def __getitem__(self, key) -> 'LapTuple':
    vs = jax.vmap(lambda x, k: x[k], in_axes=(0, None), out_axes=(0))
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
      return LapTuple(v, g, l, spars)

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
      return lap_mul(self, x)
    if isinstance(x, float) or isinstance(x, int):
      return LapTuple(self.value*x, self.grad*x, self.lap*x, self.spars)
    if isinstance(x, jnp.ndarray):
      vshape = jnp.broadcast_shapes(self.value.shape, x.shape)
      spars = deepcopy(self.spars)
      grad = spars.broadcast_dim(vshape, self.grad)
      vm = jax.vmap(jnp.multiply, (0, None), 0)
      return LapTuple(self.value*x, vm(grad, x), self.lap*x, spars)

  def __rmul__(self, x: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    return self*x

  def __truediv__(self, x: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    if isinstance(x, LapTuple):
      return lap_mul(self, reciprocal_withL(x))
    if isinstance(x, float) or isinstance(x, int):
      return LapTuple(self.value/x, self.grad/x, self.lap/x, self.spars)
    if isinstance(x, jnp.ndarray):
      vshape = jnp.broadcast_shapes(self.value.shape, x.shape)
      spars = deepcopy(self.spars)
      grad = spars.broadcast_dim(vshape, self.grad)
      vd = jax.vmap(jnp.divide, (0, None))
      return LapTuple(self.value/x, vd(grad, x), self.lap/x, spars)

  def __rtruediv__(self, x: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    # x could't be LapTuple, or __truediv__ will be called.
    return reciprocal_withL(self) * x

  def __pow__(self, p: Union[int, float, jnp.ndarray, 'LapTuple']) -> 'LapTuple':
    from lapjax import numpy as my_jnp
    return my_jnp.power(LapTuple(self.value, self.grad, self.lap, self.spars), p)

  def __str__(self) -> str:
    return 'LapTuple(\n  Value: \n{}\n  Grad: \n{} \n  Laplacian: \n{}\n)'.format(self.value, self.grad, self.lap)

  def __eq__(self, x):
    r_val = x.value if isinstance(x, LapTuple) else x
    return self.value == r_val

  def __ne__(self, x):
    r_val = x.value if isinstance(x, LapTuple) else x
    return self.value != r_val

  def __lt__(self, x):
    r_val = x.value if isinstance(x, LapTuple) else x
    return self.value < r_val

  def __gt__(self, x):
    r_val = x.value if isinstance(x, LapTuple) else x
    return self.value > r_val

  def __le__(self, x):
    r_val = x.value if isinstance(x, LapTuple) else x
    return self.value <= r_val

  def __ge__(self, x):
    r_val = x.value if isinstance(x, LapTuple) else x
    return self.value >= r_val
  
  def __len__(self):
    # TODO: This len function behavior need to be discussed in the future
    # Now, to be compatible with existing packages, we have to define the
    # len as the value array
    return len(self.value)

  @classmethod
  def identical(cls, x, y) -> bool:
    ne_fg = (x.value != y.value).any() + (x.grad != y.grad).any() + \
            (x.lap != y.lap).any() + (x.spars != y.spars)
    return not ne_fg

  def __identical(self, y) -> bool:
    return LapTuple.identical(self, y)

def reciprocal_withL(input_x: LapTuple) -> LapTuple:
  x, x_nabla, x_nabla2 = input_x.to_tuple()
  y = 1/x
  y_nabla = -x_nabla/(x[None]**2)
  y_nabla2 = -x_nabla2/(x**2) + \
             2 * jnp.sum(x_nabla ** 2, axis=0) * (y**3)
  return LapTuple(y, y_nabla, y_nabla2, input_x.spars)

def lap_mul(x: LapTuple, y: LapTuple) -> LapTuple:
  v = x.value * y.value
  assert x.spars.get_id() == y.spars.get_id()
  spars, gs = sutils.broadcast_spars([x.spars, y.spars],
                                     [x.grad, y.grad])
  g = gs[0] * y.value[None] + gs[1] * x.value[None]
  l = 2 * jnp.sum(gs[0] * gs[1], axis=0) + y.lap * x.value + x.lap * y.value
  return LapTuple(v, g, l, spars)

from lapjax.lapsrc import sparsutils as sutils

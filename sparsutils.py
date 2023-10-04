from typing import Tuple, Sequence, Union, Optional, Any, Callable
from copy import deepcopy
import heapq  

import jax
import jax.numpy as jnp
from jax._src.api_util import _ensure_index_tuple
from jax._src.numpy.lax_numpy import _Indexer
from lapjax.axis_utils import (
  AX_MAP, SHAPE, S_AXES, 
  map_axis_from_shape, reduce_axis,
)  
from lapjax.func_utils import lap_print, get_name
from lapjax.axis_utils import mod_axis, get_op_axis, merge_neg
from lapjax.laptuple import LapTuple
from lapjax.laputils import check_single_args
from lapjax.sparsinfo import SparsInfo, InputInfo, SparsTuple

def tuple2spars(flat_tups: Tuple[Tuple[int]]) -> SparsInfo:
  """Given a tuplized SparsInfo, return corresponding SparsInfo.

  Args:
      flat_tups (Tuple[Tuple[int]]): a tuple return by 
      `SparsInfo.to_tuple()`.
  """
  inputinfo = InputInfo(flat_tups[0][0], flat_tups[0][1])
  tups = [SparsTuple(*w) for w in flat_tups[1:]]
  return SparsInfo(inputinfo, tups)

def _transpose_map(a: LapTuple, 
                  axes: Optional[Sequence[int]] = None
                  ) -> AX_MAP:
  axes_ = list(range(a.ndim)[::-1]) if axes is None else axes
  axes = mod_axis(axes_, a.ndim)
  return {v: i for i, v in enumerate(axes)}

def _swapaxes_map(a: LapTuple, axis1: int, axis2: int) -> AX_MAP:
  axis1, axis2 = axis1 % a.ndim, axis2 % a.ndim
  dic = {w: w for w in range(a.ndim)}
  if axis1 != axis2:
    dic.update({axis1:axis2, axis2:axis1})
  return dic

def _squeeze_map(a: LapTuple, 
                axis: Optional[Union[int, Tuple[int, ...]]] = None,
                ) -> AX_MAP:
  if axis is not None:
    axis = _ensure_index_tuple(axis)
  else:
    axis = tuple(i for i, d in enumerate(a.shape) if d == 1)
  # these axis will be discarded, while the rest is decrease
  ax_map = {w:None if w in axis else w for w in range(a.ndim)}
  
  return reduce_axis(ax_map)

def _expand_dims_map(a: LapTuple, 
                    axis: Union[int, Sequence[int]],
                    ) -> AX_MAP:
  axis = _ensure_index_tuple(axis)
  # new axis will never gonna split the gradient.
  ndim_out = a.ndim + len(axis)
  axis = mod_axis(axis, ndim_out) # insert in this place.
  map_to = [w for w in range(ndim_out) if w not in axis]
  return {i: w for i, w in enumerate(map_to)}

def _tile_map (a: LapTuple, reps: Union[int, Sequence[int]]) -> AX_MAP:
  try:
    iter(reps)  # type: ignore[arg-type]
  except TypeError:
    reps_tup = (reps,)
  else:
    reps_tup = tuple(reps)
  # obtain the resulting axis mapping
  A_shape = (None,) * (len(reps_tup) - a.ndim) + tuple(range(a.ndim))
  reps_tup = (1,) * (len(A_shape) - len(reps_tup)) + reps_tup
  return {w: i if reps_tup[i]==1 else None for 
            i, w in enumerate(A_shape) if w is not None}

# _compute_newshape is aborted in jax==0.4.16
# we have to define it by ourselves
def _compute_newshape(a: LapTuple, newshape: SHAPE):
  """Fixes a -1 value in newshape, if present."""
  try: 
    iter(newshape)
  except: 
    newshape = (newshape,)

  assert len([w for w in newshape if w < 0]) <= 1, \
    f"There could at most be one negative value in shape, got {newshape}."
  newsize = 1
  # we remove the Poly check in the original function
  for sp in newshape:
    newsize *= sp
  
  return tuple(d if d != -1 else a.size // -newsize for d in newshape)

def _reshape_map(a: LapTuple, *args: Any, order: str = "C") -> AX_MAP:
  assert order == "C", "When calling `reshape`, LapJAX only supports order 'C'."
  newshape = _compute_newshape(a, args[0] if len(args) == 1 else args)
  oldshape = a.shape
  # obtain the axis transform from shapes.
  ax_map = map_axis_from_shape(oldshape, newshape)
  return ax_map

def _split_map(ary: LapTuple, indices_or_sections, axis: int = 0) -> AX_MAP:
  del indices_or_sections
  # TODO: update for grange split support
  axis = axis % ary.ndim
  ax_map = {w:w if w != axis else None for w in range(ary.ndim)}
  return ax_map

def _get_matrixdot_axis_map (fname, n1, n2) -> Tuple[AX_MAP, AX_MAP]:
  mg_axis = (n1-1, max(0, n2-2))
  # Obtain the axis mapping before and after applying f
  if fname == get_name(jnp.matmul):
    pre_dim = max(n2 - n1, 0)
    ax_map1 = {w: w+ pre_dim for w in range(n1-1)}
    pre_dim = max(n1 - n2, 0)
    ax_map2 = {w: w+ pre_dim for w in range(n2)}

  elif fname == get_name(jnp.dot):
    ax_map1 = {w:w for w in range(n1-1)}  
    ax_map2 = {w: w + n1 - 1 for w in range(n2)}
    ax_map2[n2-1] -= 1  # after the merged axis

  ax_map1[mg_axis[0]] = None
  ax_map2[mg_axis[1]] = None
  
  return ax_map1, ax_map2

def get_axis_map (fname, *args, **kwargs) -> AX_MAP:
  """Obtain the axis mapping for Merging functions and 
    Linear functions (except split, concatenate, and reshape).

  Return a axis mapping dict indicating the change on axes.

  If an axis s1 is swapped to axis s2, then s1: s2 exists in the dict;
  If an axis s1 is merged or squeezed, s1: None exists in the dict.
  Outside Call ensures that lap_counter(kwargs) == 0, and 
    isinstance(args[0], LapTuple).
  Notice that ALL axis will appear in dict.keys().

  Examples:
  ```
  a.shape == (4,5,6,7)
  jnp.sum(a, axis = (0, 2))
  >>>
  dict = {0: None, 1: 0, 2: None, 3: 1}
  
  jnp.sum(a, axis = (0, 2), keepdims = True)
  >>>
  dict = {0: None, 1: 1, 2: None, 3: 3}

  jnp.transpose(a, (2,0,1,3))
  >>>
  dict = {2: 0, 0: 1, 1: 2, 3: 3}
  ```

  """
  if fname in get_name([
    # Linear functions
    jnp.sum, jnp.mean,  
    jnp.repeat,
    # Merging functions
    jnp.prod, jnp.linalg.norm,
    # Custommized functions
    jax.nn.logsumexp,
    jnp.min, jnp.max, jnp.amin, jnp.amax,
  ]):
    # force keyword arguments passing.
    check_single_args(fname, args)
    axis = get_op_axis(*args, **kwargs)[0]
    keepdims = kwargs.get("keepdims", False)
    ax_map = {w: None if w in axis else w for w in range(args[0].ndim)}
    if not keepdims and fname != get_name(jnp.repeat):
      ax_map = reduce_axis(ax_map)

  elif fname == get_name(jax.nn.softmax):
    assert "axis" in kwargs.keys()
    axis = mod_axis(kwargs["axis"], args[0].ndim)
    if type(axis) not in [list, tuple]:
      axis = (axis,)
    ax_map = {w: None if w in axis else w for w in range(args[0].ndim)}

  elif fname == get_name(jnp.transpose):
    ax_map = _transpose_map(*args, **kwargs)
    
  elif fname == get_name(jnp.swapaxes):
    ax_map = _swapaxes_map(*args, **kwargs)
  
  elif fname == get_name(jnp.squeeze):
    ax_map = _squeeze_map(*args, **kwargs)
  
  elif fname == get_name(jnp.expand_dims):
    ax_map = _expand_dims_map(*args, **kwargs)
  
  elif fname in get_name([jnp.triu, jnp.tril]):
    ax_map = {w:w for w in range(args[0].ndim)} # remains
  
  elif fname == get_name(jnp.tile):
    ax_map = _tile_map(*args, **kwargs)

  elif fname == get_name(jnp.linalg.slogdet):
    ndim = args[0].ndim
    ax_map = {w:w for w in range(ndim - 2)}
    ax_map.update({ndim-2:None, ndim-1:None})

  elif fname == get_name(jnp.reshape):
    ax_map = _reshape_map(*args, **kwargs)
    
  elif fname in get_name([jnp.split, jnp.array_split]):
    ax_map = _split_map(*args, **kwargs)
  
  lap_print(f"  axes map: {ax_map}")
  return ax_map

class _shapeQ(object):
  """Heapq element used for `get_common_spars`.
  """

  def __init__(self, idx: int, s: int, e:int, size: int, splits: S_AXES):
    self.idx = idx
    self.s, self.e, self.size = s, e, size
    self.splits = splits
  
  def __lt__ (self, x: "_shapeQ"):
    return self.idx < x.idx if self.size == x.size else self.size < x.size

  def __str__(self) -> str:
    return f"size: {self.size}, idx: {self.idx}, s: {self.s}, e: {self.e}, split: {self.splits}"

  def __repr__(self) -> str:
    return str(self)

def all_equal (target: Sequence[S_AXES]):
  a = target[0]
  for w in target:
    if a != w:
      return False
  return True

def get_common_spars (l_spars: Sequence[SparsInfo],
                      l_gshape: Sequence[SHAPE],
                      ) -> Tuple[SparsInfo, S_AXES]:
  """Find the longest sparsity split shared by SparsInfo in `l_spars`.
  This is used when multiple LapTuples are merged, e.g. add and concat.

  Args:
      l_spars (Sequence[SparsInfo]): List of SparsInfo.
      l_gshape (Sequence[SHAPE]): List of corresponding grad array.

  Returns:
      Tuple[SparsInfo, S_AXES]: Return the resulting SparsInfo,
        as well as the sparse axes that could be maintained.
  """

  assert len(l_spars) == len(l_gshape)
  info = l_spars[0].input
  irange = l_spars[0].tups[0].irange

  for spars in l_spars:
    # TODO: support multiple sparstuple
    assert len(spars.tups) == 1
    tup = spars.tups[0]
    assert info.getID() == spars.input.getID(), \
      f"InputInfo mismatch. Expect {info}, but got {spars.input}."
    assert tup.irange == irange,  \
           (f"Laplacian dependency mismatch. Expect irange: {irange}, "
           f" but got irange {tup.irange}.")
           
  # The axis to be concatenated has already been discarded.
  h = [_shapeQ(i, 0, 0, 1, w.tups[0].splits) for i, w in enumerate(l_spars)]
  splits, save_axis, gdim = [], [], 1
  while min([w.s - len(w.splits) for w in h]) < 0:
    # construct the heapq
    max_size = 0
    while h[0].size != max_size:  # stop condition must be equal
      q = heapq.heappop(h)
      n_size = -q.splits[q.e] if q.splits[q.e] < 0 else \
                l_gshape[q.idx][q.splits[q.e] + 1]
      q.size *= n_size
      max_size = max(max_size, q.size)
      q.e += 1
      heapq.heappush(h, q)

    # print(f"---Reach equaling gradient size.\n    {h}")
    # reach a stopping time with all size equal, check whether to save
    if all_equal([q.splits[q.s: q.e] for q in h]) and h[0].e - h[0].s == 1:
      splits.append(h[0].splits[h[0].s])
      if splits[-1] >= 0:
        save_axis.append(splits[-1])
      else:
        gdim *= -splits[-1]
    else:
      splits.append(-h[0].size) # becomes gradient dimension
      gdim *= h[0].size

    # update heapq info
    for q in h:
      q.size, q.s, q.e = 1, q.e, q.e

  spars = SparsTuple(irange, (0, gdim), merge_neg(tuple(splits)))
  return SparsInfo(inputinfo=info, tups=[spars]), save_axis

def shortcut_for_discard (f, *args, **kwargs) -> Union[LapTuple, None]:
  """This function tries to operate on non-discard axes first.
  This helps reduce the size of the grad array.
  `f` should be splitable, i.e. the operation order among axes will not
    change the result, e.g. mean, sum, max, min.
  If shortcut fails, return None; 
  Otherwise, return the resulting LapTuple.
  """

  if "where" in kwargs.keys() or "initial" in kwargs.keys():
    return None

  fname = get_name(f)
  op_axis, _, p_kwargs = get_op_axis(*args, **kwargs)
  check_single_args(fname, args)
  array: LapTuple = args[0]
  
  try:
    spars_axes = set(sum([w.splits for w in array.spars.tups]))
  except:
    spars_axes = {}
  dis_axis = tuple([w for w in op_axis if w in spars_axes])
  remain_axis = tuple([w for w in op_axis if \
    w not in spars_axes and array.shape[w] != 1])
  if len(dis_axis) == 0 or len(remain_axis) == 0:
    return None

  p_kwargs.update({"axis": remain_axis, "keepdims": True})
  import lapjax.numpy as my_jnp
  f = getattr(my_jnp, fname)
  return f(f(*args, **p_kwargs), **kwargs)

def align_irange (l_spars: Sequence[SparsInfo],
                  l_grad: Sequence[jnp.ndarray]
                  ) -> Tuple[Sequence[SparsInfo], Sequence[jnp.ndarray]]:
  """Make sure the irange in l_spars is aligned with each other.
  """
  iranges = [w.get_irange() for w in l_spars]
  irange = (min([w[0] for w in iranges]),
            max([w[1] for w in iranges]))
  l_grad = [spars.set_irange(irange, g) for spars, g in zip(l_spars, l_grad)]
  return l_spars, l_grad

def broadcast_spars(l_spars: Sequence[SparsInfo],
                    l_grad: Sequence[jnp.ndarray]
                    ) -> Tuple[SparsInfo, Sequence[jnp.ndarray]]:
  """Change the sparsity according to the broadcast rule.
  Arrays in `l_grad` is the array to be broadcast and merged.

  Returns:
      Tuple[SparsInfo, jnp.ndarray]: The resulting SparsInfo and grad_out.
  """

  l_spars = deepcopy(l_spars)
  l_spars, l_grad = align_irange(l_spars, l_grad)

  vshape = l_grad[0].shape[1:]
  for grad in l_grad:
    vshape = jnp.broadcast_shapes(vshape, grad.shape[1:])
  
  l_grad = [spars.broadcast_dim(vshape, g) for 
              spars, g in zip(l_spars, l_grad)] # change spars and grad.
  # upon here, all grad will have identical shape[1:]
  
  res_spars, save_axis = get_common_spars(l_spars, [w.shape for w in l_grad])
  dis_axis = set([w for w in range(len(vshape)) if w not in save_axis])
  l_grad = [spars.discard(dis_axis, g) for 
              spars, g in zip(l_spars, l_grad)]
  assert all_equal([w.shape for w in l_grad])
  

  return res_spars, l_grad

def matrix_spars (lap_idx: int, spars:SparsInfo, f: Callable,
                  g1: jnp.ndarray, g2: jnp.ndarray,
                  ) -> Tuple[SparsInfo, jnp.ndarray]:  
  """Changed SparsInfo when `dot` and `matmul` is called.

  Args:
      lap_idx (int): which input is LapTuple, left (0) or right (1).
      spars (SparsInfo): The SparsInfo of the LapTuple.
      f (Callable): The partial function to be called, with correct `f.__name__`.
      g1 (jnp.ndarray): The LHS input of f (can be grad).
      g2 (jnp.ndarray): The RHG input of f (can be grad).

  Returns:
      Tuple[SparsInfo, jnp.ndarray]: The resulting SparsInfo and grad_out.
  """

  spars = deepcopy(spars)
  ax_map = _get_matrixdot_axis_map(f.__name__, 
                                  n1=g1.ndim - (1-lap_idx),
                                  n2=g2.ndim - lap_idx)[lap_idx]
  # discard merging axis
  if lap_idx == 0:  # The left one is LapTuple
    g1 = spars.discard({g1.ndim - 2}, g1)
    out = jax.vmap(f, in_axes=(0,None), out_axes=0)(g1, g2)
  else:
    g2 = spars.discard({max(g2.ndim - 3, 0)}, g2)
    out = jax.vmap(f, in_axes=(None,0), out_axes=0)(g1, g2)
  
  spars.swap_axis({k:v for k,v in ax_map.items() if v != None})
  return spars, out

def getitem_map (indexer: _Indexer, ndim: int) -> AX_MAP:
  """Given the __getitem__ indexer, obtain the ax_map on spars.
  """

  assert len(indexer.dnums.offset_dims) + len(indexer.dnums.collapsed_slice_dims) == ndim
  main_axes = [w for w in range(ndim) if w not in indexer.dnums.collapsed_slice_dims]
  ax_map = {u:v for u,v in zip(main_axes, indexer.dnums.offset_dims)}
  ax_map.update({w: None for w in indexer.dnums.collapsed_slice_dims})
  ax_map.update({w: None for w in indexer.dnums.start_index_map})

  return ax_map

def take_diag(x: jnp.ndarray, saxis: int, raxis: int) -> jnp.ndarray:
  """Take the diagonal term of x along (saxis, raxis) and put on saxis.
  Notice that the dimension of raxis is set to 1.
  """
  x = x[..., None, None].swapaxes(-2, saxis).swapaxes(-1, raxis)
  shape = x.shape[:-2]
  _x: jnp.ndarray = x.reshape((-1,)+x.shape[-2:])
  _x = jax.vmap(jnp.diag)(_x).reshape(shape + (-1,))
  return _x.swapaxes(saxis, -1)[..., 0]

def anti_diag(x: jnp.ndarray, gaxis: int, axes: Sequence[int]) -> jnp.ndarray:
  """Switch the gradient axis `gaxis` back to all axis in `axes`.
  If 2 axes exist in `axes`, the result should be a diagonal matrix.
  """
  assert len(axes) <= 2
  if len(axes) == 0:
    return x
  axes = tuple(set(axes))
  for i in axes:
    assert x.shape[i] == 1
  if len(axes) == 1:
    return x.swapaxes(gaxis, axes[0])
  else:
    x = x[..., None].swapaxes(gaxis, -1)
    shape = x.shape[:-1]
    _x: jnp.ndarray = jax.vmap(jnp.diag)(x.reshape(-1, x.shape[-1]))
    _x = _x.reshape(*shape, x.shape[-1], x.shape[-1])
    return _x.swapaxes(-2, axes[0]).swapaxes(-1, axes[1])[..., 0, 0]

def sum_matrix_grad(f: Callable,
                    l_spars: Sequence[SparsInfo], 
                    l_grad: Sequence[jnp.ndarray],
                    ) -> Union[None, jnp.ndarray]:
  """Accelerate the sum along grad axis of `f(*l_grad)`.
  If not implemented accelerated patterns are found, return None.
  Caller should handle this possible return. 

  Accelerated Patterns:
    This subroutine leverages the linearity of `matmul` and `dot`.
    Specifically, it tries to move the dimension from a sparse axis
      to the grad axis before calling `f`, which largely reduce shape. 
    Since eventually we sum over all gradient axes, the result remains
      after we move axes, so long as the following conditions hold:
    1.  The gradient shapes (obtained from SparsTuple) of 2 inputs
        satisfies that one is strictly a further split of the other.
        For example, shape (3, 4, 5) and shape (12, 5) satisfies (
        and we call (3, 4, 5) the fine-grained shape), while
        shape (12, 5) and shape (3, 20) do not. 
    2.  For both arrays, every sparse axis correspond at most to ONE axis
        in the fine-grained gradient shape. For example, for shape (12, 5),
        the first axis must not belong to a sparse axis.
    3.  The merged axis, if both are sparse, should correspond to the
        same axis in the grad shape (either sparse is supported).
        
  Args:
      f (Callable): The operating function, should be `matmul` or `dot`.
      l_spars (Sequence[SparsInfo]): SparsInfo of each grad array.
      l_grad (Sequence[jnp.ndarray]): The corresponding grad array sequence.
  """
  
  assert len(l_spars) == len(l_grad) == 2
  l_spars, l_grad = align_irange(deepcopy(l_spars), l_grad)
  
  n1, n2 = l_grad[0].ndim - 1, l_grad[1].ndim - 1 # the ndim of v
  assert n1 >= 1 and n2 >= 1  # not scalar
  l_splits = [w.tups[0].splits for w in l_spars]
  
  # 1. Check if one grad shape is more fine-grained than the other.
  # If not, the sparse behavior can be complex,
  # and we directly return None to stop the subroutine.
  # Otherwise, the fine-grained grad shape and its idx is recorded.
  gshapes = [s.get_gshape(g) for s, g in zip(l_spars, l_grad)]
  ax_map = map_axis_from_shape(*gshapes)
  if max([len(k) for k in ax_map.keys()]) == 1:
    idx, gshape, gdim = 1, gshapes[1], len(gshapes[1])
  elif max([len(v) for v in ax_map.values()]) == 1:
    idx, gshape, gdim = 0, gshapes[0], len(gshapes[0]) 
  else:
    return None
  
  # 2. Check is all sparse axes in splits correspond only to 
  # a single axis in `gshape`. If not, stop this subroutine.
  mg_axis = (n1-1, max(0, n2-2))
  grad_maps: Sequence[dict] = []
  for splits, shape in zip(l_splits, gshapes):
    maps = map_axis_from_shape(shape, gshape)
    assert max([len(k) for k in maps.keys()]) == 1
    # maps is {sparse axis: grad axis}
    maps = {splits[k[0]]: v for k,v in maps.items() if splits[k[0]] >= 0} 
    if len(maps) and max([len(v) for v in maps.values()]) > 1:
      return None # Condition 2 breaks
    grad_maps.append({k: v[0] for k,v in maps.items()})

  # 3. In addition, check the merged axis to see their sparsity.
  # If both are sparse but are different axes, stop the subroutine.
  mg_grad_axis = [grad_maps[i].get(axis, None) for 
                  i, axis in enumerate(mg_axis)]
  if mg_grad_axis[0] is not None and mg_grad_axis[1] is not None and \
    mg_grad_axis[0] != mg_grad_axis[1]:
      return None  
  
  # Till here, all conditions are satisfied, and we move the sparse axis
  # to the gradient to speed up the computation of f.
  
  ax_maps = _get_matrixdot_axis_map(f.__name__, n1, n2)
  # swap_map is {grad axis: (out axes)}. Used to switch grad axis back.
  swap_map = {i: [] for i in range(gdim)}  
  for i, (splits, ax_map) in enumerate(zip(l_splits, ax_maps)):
    neg_shape = tuple([-w for w in splits if w < 0])
    grad = l_grad[i].reshape(neg_shape + l_grad[i].shape[1:])
    
    exp_map = {w:i for i, w in enumerate(splits) if w >= 0}
    grad = jnp.expand_dims(grad, tuple(exp_map.values()))
    for k,v in exp_map.items():
      grad = grad.swapaxes(v,k+len(splits))
      if ax_map[k] is not None: # exist in out shape, i.e. not merged
        swap_map[grad_maps[i][k]].append(ax_map[k]+gdim)
    l_grad[i] = grad.reshape(gshape + grad.shape[len(splits):])
    
  # Now, grad arrays are shaped as gshape + vshape.
  # Corner case: we take the diagonal for the possible merged axis,
  # if only either grad is sparse along the merged axis.
  if int(mg_grad_axis[0] is None) + int(mg_grad_axis[1] is None) == 1:
    idx = int(mg_grad_axis[1] is None)
    axis = mg_grad_axis[1-idx]
    l_grad[idx] = take_diag(l_grad[idx], axis, mg_axis[idx] + gdim)
  
  # Compress axes in shape, in preparation for vmap.
  l_grad = [w.reshape(-1, *w.shape[gdim:]) for w in l_grad]
  out = jax.vmap(f)(*l_grad)
  out = jnp.reshape(out, gshape + out.shape[1:])

  # Swap axes from grad shape back to array, and sum over grad axes.
  for k, v in swap_map.items():
    out = anti_diag(out, k, v)
  return jnp.sum(out, axis = tuple([w for w in range(gdim)]))

def check_irange_continuity (
    arrays: Tuple[Union[jnp.ndarray, LapTuple]],
    axis: int
  ) -> Tuple[bool, Tuple[int]]:
  """Check whether concatenating along `axis` does not change splits
  but only expands irange.
  This means that:
  1. The dependence on inputs are sequential and continuous.
  2. The splited irange length of each LapTuple is identical. 
  If the above two rules are satisfied, 

  Args:
      arrays (Tuple[Union[jnp.ndarray, LapTuple]]): Array sequence to
        be concatenated.
      axis (int): The operating axis. Already set range to [0, ndim).

  """
  continuity, left, right = True, -1, -1
  rest_size = -1  # the size of each 'split' after diving along axis.
  for array in arrays:
    if not isinstance(array, LapTuple):
      continuity = False
      continue
    tup = array.spars.tups[0]
    l, r = tup.irange
    split_irange_size = r - l
    if axis in tup.splits:
      assert split_irange_size % array.shape[axis] == 0
      split_irange_size /= array.shape[axis]
      continuity = continuity if axis == tup.splits[0] else False
    else:
      continuity = continuity if array.shape[axis] == 1 else False

    if left == -1:  # initial entry, set variables
      left, right, rest_size = l, r, split_irange_size
    else:
      # Continuity not satisfied. Set to False and maintain dependence.
      if not continuity or right != l or split_irange_size != rest_size:
        continuity = False
        left, right = min(left, l), max(right, r)
      else:
        right = tup.irange[1]
  return continuity, (left, right)

def expand_irange (l_spars: Sequence[SparsInfo],
                   op_axis: int,
                   irange: Tuple[int],
                   ) -> Tuple[Sequence[SparsInfo]]:
  """Expand irange of spars in `l_spas`, and remove
    the leading op_axis if it is.
    This axis will be prepended outside this function.

  Returns:
      Tuple[Sequence[SparsInfo]]: The resuling SparsInfo list.
  """
  o_spars = []
  for spars in l_spars:
    splits = spars.tups[0].splits
    splits = splits if splits[0] != op_axis else splits[1:]
    o_spars.append(
      SparsInfo(inputinfo=spars.input,
                tups=[SparsTuple(irange, spars.tups[0].grange, splits)]))
  return o_spars

def concat_sparsity (
    arrays: Sequence[Union[LapTuple, jnp.ndarray]],
    op_axis: int,
  ) -> Tuple[Sequence[Union[LapTuple, jnp.ndarray]], SparsInfo]:
  """Obtain the SparsTuple after concating `arrays` along axis `op_axis`
  First, check whether the input dependence is continuously splited to
  arrays along the axis (finished by `check_irange_continuity`).
  If True, we will not discard the operating axis, but prepend it before 
    return. This means that the sparsity along this axis is maintaiend.
  Otherwise, we need to discard this axis, and we set the input dependence
    of all inputs to the total irange returned by `check_irange_continuity`.

  Returns:
      List of resulting arrays, and the SparsTuple for final output array.
  """
  op_axis = op_axis % arrays[0].ndim
  maintain_axis, irange = check_irange_continuity(arrays, op_axis) 
  if maintain_axis:
    l_spars = expand_irange([w.spars for w in arrays], op_axis, irange)
    for array, spars in zip(arrays, l_spars):
      array.spars = spars
  else:
    arrays = [w.discard({op_axis}) if isinstance(w, LapTuple) else w for w in arrays]
    arrays = [w.set_irange(irange) if isinstance(w, LapTuple) else w for w in arrays]

  # Find the maximum common SparsTuple FOR THE REST SPARSITY.
  spars, save_axis = get_common_spars(
    l_spars=[w.spars for w in arrays if isinstance(w, LapTuple)],
    l_gshape=[w.grad.shape for w in arrays if isinstance(w, LapTuple)]
  )

  discard_axis = set([w for w in range(arrays[0].ndim) if w not in save_axis])
  arrays = [w.discard(discard_axis) if isinstance(w, LapTuple) else \
            LapTuple(w, spars=spars) for w in arrays]
  
  # Append the leading irange axis to splits, if spasity is maintained.
  if maintain_axis != False:
    spars.tups[0].splits = (op_axis,) + spars.tups[0].splits
  
  return arrays, spars
from typing import Tuple, Sequence, Set
import time
from copy import deepcopy

import jax
import jax.numpy as jnp
from lapjax.func_utils import lap_print
from lapjax.axis_utils import AX_MAP, SHAPE, merge_neg, parse_splits

from lapjax.func_utils import lap_print

class InputInfo (object):
  def __init__(self, size: int, id: int = None) -> None:
    if id is None:
      self.id = int(time.time())
      lap_print("Specifying new inputs for forward laplacian computation ...\n",
            f"  Input ID: {self.id}, Length: {size}")
    else:
      self.id = id
    self.size = size
  
  def getSize(self) -> int:
    return self.size

  def getID(self) -> int:
    return self.id

  def __str__(self) -> str:
    return f"ID{self.id} (Size:{self.size})"

  def to_tuple(self) -> Tuple[int]:
    return (self.size, self.id)


class SparsTuple(object):
  """
    A tuple as `(irange, grange, splits ...)`
    where:
    `irange`: a binary tuple representing the range of input that 
            this tuple depends on.
    `grad_dim`: a binary tuple representing the range of gradient
                that this tuple specify.
    `splits`: several dimension index (with neg value means grad dim),
              where the order specifies the sequence to divide sparsity.
  """
  def __init__(self, irange: Tuple[int],
                     grange: Tuple[int], 
                     splits: Tuple[int]) -> None:
    assert len(irange) == len(grange) == 2
    self.irange = irange
    self.grange = grange
    self.splits = splits
  
  def get_idim(self) -> int:
    return self.irange[1] - self.irange[0]
  
  def get_gdim(self) -> int:
    return self.grange[1] - self.grange[0]

  def __repr__(self) -> str:
    return str(self)

  def __str__(self) -> str:
    return f"({self.irange}, {self.grange}, {self.splits})"

  def to_tuple(self) -> Tuple[int]:
    return self.irange, self.grange, self.splits

  def swap_axis (self, maps: AX_MAP) -> None:
    # maps should contain all axis
    self.splits = tuple([
      w if w < 0 else maps[w] for w in self.splits])

  def swap_tuple_axis (self, maps: AX_MAP) -> None:
    self.splits = parse_splits(self.splits, maps)[0]
    

  def get_discard_from_map (self, ax_map: AX_MAP) -> Set[int]:
    return parse_splits(self.splits, ax_map)[1]

  def discard (self, op_axis: Set[int], grad: jnp.ndarray) -> jnp.ndarray:
    vshape, neg_shape = grad.shape[1:], [-w for w in self.splits if w < 0]
    assert grad.shape[0] == self.get_gdim()
    
    pointer = len(neg_shape)  # how many grad dim before
    gdim = self.get_gdim()

    out_splits = []
    for axis in self.splits[::-1]:
      if axis not in op_axis: # not discard, i.e. grad axis or unchanged
        out_splits.append(axis) # reverse at last
        pointer = pointer -1 if axis < 0 else pointer
        continue

      size = grad.shape[axis+1] # the size of discard dimension
      _grad = grad.swapaxes(-1, axis+1) # set discard axis to last
      # print(f"    discarding axis {axis} with size {size}.")

      vd = jax.jit(jax.vmap(lambda x: jnp.diag(x)))
      _grad: jnp.ndarray = vd(_grad.reshape((-1, size))).reshape(
              _grad.shape[:-1] + (size, size)) # (gdim, rest, size, size)
      # print(f"    shape after anti-diag: {_grad.shape}, neg_shape: {tuple(neg_shape)}")
      
      # return the original axis, shape = (gdim, value.shape, size)
      _grad = _grad.swapaxes(-2, axis+1).reshape(tuple(neg_shape)+vshape+(size,)) 

      trans = list(range(_grad.ndim))[:-1]
      trans.insert(pointer, len(trans))
      
      gdim *= size
      neg_shape.insert(pointer, size)
      out_splits.append(-size)
      grad = _grad.transpose(tuple(trans)).reshape((gdim,) + vshape)
      # print(f"    Solved. gdim: {gdim}, new grad shape: {grad.shape}")
      
    self.splits = merge_neg(tuple(out_splits[::-1]))
    return grad
    

class SparsInfo (object):
  """List of Tuples. 
  Each tuple represent a gradient sparsity allocation.
  All tuples correspond to a single InputInfo.

  #### Input info
    A InputInfo indicating which input these tuples are pointing.


  TODO: add operations on SparsTuple.
  """

  def __init__(self, 
               inputinfo: InputInfo,
               tups: Sequence[SparsTuple] = None,
               *,
               is_input = False,
               ndim = 1,
               ) -> None:
    """TODO: add documentation.
    """
    self.input = inputinfo
    if is_input:  # This sparsinfo is about the initial input.
      # split in dim 0, then in the gradient dim (-1)
      # ndim is only useful when is_input is True
      self.tups = [SparsTuple(irange=(0, inputinfo.getSize()), 
                              grange=(0, 1),
                              splits=tuple(range(ndim)))]
      return
    if tups is not None:
      self.tups = tups
    else:
      ele = SparsTuple(irange=(0, self.input.getSize()),
                       grange=(0, self.input.getSize()),
                       splits=(-self.input.getSize(),))
      
      self.tups = [ele]

  def get_id (self):
    return self.input.getID()

  def get_gdim(self):
    return sum([w.get_gdim() for w in self.tups])
  
  def get_irange(self):
    iranges = [w.irange for w in self.tups]
    l = min([w[0] for w in iranges])
    r = max([w[1] for w in iranges])
    return (l,r)

  def get_gshape (self, grad: jnp.ndarray):
    assert len(self.tups) == 1
    splits = self.tups[0].splits
    return tuple([-w if w < 0 else grad.shape[1+w] for w in splits])

  def __repr__(self) -> str:
    return str(self)

  def __str__(self) -> str:
    return f"InputInfo: {self.input}, " + \
           f"Sparsity Tuple: {self.tups}"

  def to_tuple(self) -> Tuple[Tuple[int]]:
    return (self.input.to_tuple(), ) + tuple([
      w.to_tuple() for w in self.tups
    ])

  def get_grad_split(self, grad: jnp.ndarray):
    arr_splits =  [w.grange[0] for w in self.tups][1:]
    return jnp.array_split(grad, arr_splits)

  def set_grange (self, grads: Tuple[jnp.ndarray]) -> None:
    # set the grange of self.tups according to grad list `grads`.
    import numpy as np
    g_ranges = np.cumsum([0]+[w.shape[0] for w in grads])
    for i, sparstuple in enumerate(self.tups):
      sparstuple.grange = (g_ranges[i], g_ranges[i+1])

  def discard (self, op_axis: Set[int], grad: jnp.ndarray) -> jnp.ndarray:
    """Sequential check all sparstuple, and remove axis
    in op_axis to gradient dim.
    Set the grad accordingly and return the resulting grad.

    Args:
        op_axis (Tuple[int]): The axis tuple to be discarded.
        grad (jnp.ndarray): The corresponding grad ndarray.

    Returns:
        jnp.ndarray: Resulting (longer) grad ndarray.
    """
    grads = self.get_grad_split(grad)
    grads = [w.discard(op_axis, g) for w,g in zip(self.tups, grads)]
    # TODO: consider merging tuples for further acceleration.
    self.set_grange(grads)

    return jnp.concatenate(grads, axis = 0)
  
  def check_leading_axis (self, axis: int):
    """This is used for `vmap`, `split`, and `split_array`.
    Check whether the axis is the leading axis,
    e.g. the first dimension appeared in SparsTuple. 

    Args:
        axis (int): The axis to be split.
    """
    assert len(self.tups) == 1, NotImplementedError
    return axis == self.tups[0].splits[0]

  def split_sparsity (self, axis: int, size: Sequence[int]) -> Sequence["SparsTuple"]:
    tup = self.tups[0]
    irange = tup.irange
    import numpy as np
    ipoint = np.cumsum([0] + [tup.get_idim() // sum(size) * w for w in size])
    l_irange = [(irange[0]+ipoint[i], irange[0]+ipoint[i+1]) for i in range(len(size))]
    gets = lambda t, w: t[1:] if w == 1 else t
    l_splits = [gets(tup.splits, w) for w in size]
    return [SparsInfo(deepcopy(self.input), 
                      [SparsTuple(w, tup.grange, s)]
                      ) for w,s in zip(l_irange, l_splits)]

  def set_irange (self, irange: Tuple[int], grad: jnp.ndarray) -> jnp.ndarray:
    """Set the irange of sparstuple to `irange`, and return the expanded grad.
    Notice that irange must be wider than current irange.
    """

    assert len(self.tups) == 1
    tup = self.tups[0]
    assert tup.irange[0] >= irange[0] and tup.irange[1] <= irange[1]
    if tup.irange[0] == irange[0] and tup.irange[1] == irange[1]:
      return grad
    # We need to expand the gradient dim and fill with zeros.
    # To do so, we need to discard several axis until we found that 
    # grad takes the integral idnex.
    
    lead_gdim = 1 if tup.splits[0] >=0 else -tup.splits[0]
    assert tup.get_idim() % lead_gdim == 0
    gcd_size = tup.get_idim() // lead_gdim # size of one split
    while (irange[1] - irange[0]) % gcd_size != 0 or \
          (tup.irange[0] - irange[0]) % gcd_size != 0:
      grad = tup.discard({[w for w in tup.splits if w >= 0][0]}, grad)
      lead_gdim = -tup.splits[0]
      gcd_size = tup.get_idim() // lead_gdim

    n_total = (irange[1] - irange[0]) // gcd_size
    rest_gdim = grad.shape[0] // lead_gdim
    idx = (tup.irange[0] - irange[0]) // gcd_size * rest_gdim
    _grad = jnp.zeros((n_total * rest_gdim,)+grad.shape[1:], dtype=grad.dtype)
    _grad = _grad.at[idx:idx+grad.shape[0]].set(grad)
    splits = merge_neg((-1,)+tup.splits)
    self.tups = [SparsTuple(irange=irange,
                            grange=(0,_grad.shape[0]),
                            splits=(-n_total, ) + splits[1:]
                            )]
    return _grad

  def set_dense (self, grad: jnp.ndarray, force: bool = False) -> jnp.ndarray:
    """Set the sparsinfo to dense style (full gradient) if 
    this reduces the size of grad. If `force`, set to dense anyway.

    Args:
        grad (jnp.ndarray): The corresponding grad ndarray.
        force (bool, optional): Whether force to dense. Defaults to False.

    Returns:
        jnp.ndarray: Resulting full grad ndarray, or remainging grad if
                     force is False and sparsity still helps.
    """
    if force:
      return self.discard(set([w for w in range(grad.ndim)]), grad)
    else: 
      raise NotImplementedError("Yet to implement dynamically densify.")

  def swap_axis (self, maps: AX_MAP) -> None:
    """Swaps the axis in sparstuple according to key-value in maps.
    This will not change the sparsity, and not None value should exist.

    Args:
        maps (Mapping): key is the initial axis,
                and value is the resulting axis.
    """
    [w.swap_axis(maps) for w in self.tups]

  def swap_tuple_axis (self, maps: AX_MAP) -> None:
    """Swap axis, where keys and values are axis tuple.s
    Not all axis tuple exist. Onde we find an existing key, 
    replace the key with the value.
    """
    [w.swap_tuple_axis(maps) for w in self.tups]
  
  def map_for_reshape(self, 
                      grad: jnp.ndarray, 
                      ax_map: AX_MAP) -> jnp.ndarray:
    """Change the sparsity specifically for `reshape` function.
    `grad` is the original gradient array.

    Returns:
        jnp.ndarray: The resulting grad array.
    """
    grads = self.get_grad_split(grad)
    grads = [w.discard(w.get_discard_from_map(ax_map), g) 
              for w,g in zip(self.tups, grads)]

    self.set_grange(grads)

    # swap tuple axis
    self.swap_tuple_axis(ax_map)

    return jnp.concatenate(grads, axis = 0)

  def broadcast_dim(self, ns: SHAPE, grad: jnp.ndarray) -> jnp.ndarray:
    """Broadcast SparsInfo to new shape `ns`.

    Returns:
        jnp.ndarray: The resulting grad array.
    """
    add_dim = len(ns) + 1 - grad.ndim
    for w in self.tups:
      w.splits = tuple([i if i < 0 else i + add_dim for i in w.splits])
    return jax.vmap(jnp.broadcast_to, (0, None))(grad, ns)
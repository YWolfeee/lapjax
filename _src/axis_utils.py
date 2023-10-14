"""
Several utility functions operating on axis, including
tuples, ax_map, and shapes.

Should not depend on any data structure.
"""
from typing import Tuple, Sequence, Union, Optional, Any, Set, Dict
from copy import deepcopy
import numpy as np

SHAPE = Tuple[int]
AXIS = Union[int, Sequence[int], Tuple[int]]
S_AXES = Tuple[int]
AX_MAP = Dict[Union[Tuple[int], int], Union[Tuple[int], int, None]]

def mod_axis(axis: AXIS, length: int) -> AXIS:
  # Return axis % length
  if length == 0:
    f = lambda x: None if x is None else 0
  else:
    f = lambda x: None if x is None else x % length
  if type(axis) == int:
    return f(axis)
  elif type(axis) == tuple:
    return tuple([f(w) for w in axis])
  elif type(axis) == list:
    return [f(w) for w in axis]
  raise  

def get_op_axis (*args, **kwargs) -> Tuple[Set[int], tuple, dict] :
  """Return the axis tuple that the function is operating on.

  When `axis` in kwargs: extract initial input axis;
  Otherwise, ASSUME AXIS IS NONE (Should not be in args).
  
  When axis is not None, return its mod; Otherwise, return args[0].shape
  
  Further, remove the axis information from args and kwargs.
  Return = (set(axis), new_args, new_kwargs)
  """
  p_args, p_kwargs = deepcopy(args), deepcopy(kwargs)
  if 'axis' in p_kwargs.keys():
    axis = p_args[0].shape if p_kwargs['axis'] is None else p_kwargs['axis']
    p_kwargs.pop('axis')
  else:  # axis is None, applying on all dimensions
    axis = list(range(len(p_args[0].shape)))
  if type(axis) not in [list, tuple]:
    axis = [axis]
  if len(axis) == 0:
    return {0}, p_args, p_kwargs
  axis = set(mod_axis(axis, p_args[0].ndim))
  
  return axis, p_args, p_kwargs

def merge_neg (splits: S_AXES) -> S_AXES:
    """Merge continuous negative dim.
    """
    res, i, j, length = [], 0, 0, len(splits)
    while i < length:
      if splits[i] >= 0:
        res.append(splits[i])
        i += 1
      else:
        j = i + 1
        while j < length and splits[j] < 0:
          j += 1
        res.append(-int(np.prod([-w for w in splits[i:j]])))
        i = j
    return tuple(res)

def map_axis_from_shape (old: SHAPE, 
                         new: SHAPE) -> AX_MAP:
  # All axis with size 1 should never appear in SparsTuple.
  res_dic = {}
  o_i, n_i = len(old) - 1, len(new) - 1
  while o_i >= 0:
    o_p, n_p = o_i, n_i
    o_s, n_s = old[o_p], new[n_p]
    if old[o_i] == 1:
      o_i -= 1
      continue
    while o_s != n_s:
      if o_s < n_s:
        o_p -= 1
        o_s *= old[o_p]
      else:
        n_p -= 1
        n_s *= new[n_p]
    key = tuple([w for w in range(o_p, o_i+1) if old[w] != 1])
    value = tuple([w for w in range(n_p, n_i+1) if new[w] != 1])
    res_dic[key] = value
    o_i, n_i = o_p - 1, n_p - 1
  
  return res_dic

def parse_splits (splits: S_AXES, maps: AX_MAP):
    init_keys = {w[0]: w for w in maps.keys()}
    p, axes, dis = 0, [], []
    while p < len(splits):  
      if splits[p] in init_keys:
        ik = splits[p] # the initial axis
        if splits[p: p + len(init_keys[ik])] == init_keys[ik]:
          axes = axes + list(maps[init_keys[ik]])
          p += len(init_keys[ik])
          continue
      if splits[p] < 0:
        axes.append(splits[p])
      else:
        dis.append(splits[p])
      p += 1
    
    return tuple(axes), set(dis)

def reduce_axis (ax_map: AX_MAP) -> AX_MAP:
  """Check the ax_map and pick the key with value == None.
  This axis key will be removed.
  As a result, decrease the rest axis index accordingly.
  """

  no_axis = [w for w, v in ax_map.items() if v is None]
  return {w: v if v is None else v - sum([int(i<v) for i in no_axis]) for \
              w, v in ax_map.items()}

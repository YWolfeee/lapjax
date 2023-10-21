from typing import Callable

from jax import numpy as jnp

from lapjax.lapsrc.laptuple import TupType, LapTuple

#### Argument checking utilities.

def check_lapcount_args (fname, args):
  assert lap_counter(args) == 1, \
    (f"When calling `{fname}`, there should only "
    "be at most one LapTuple in position argument.")

def check_single_args (fname, args):
  assert len(args) == 1, \
    ("For the sake of LapJAX acceleration, we accept and only accept a single "
    f"position argument when calling `{fname}`. "
    "Please pass the rest as keyword arguments.")

def check_pure_kwargs (fname, kwargs):
  assert lap_counter(kwargs) == 0, \
    (f"When calling `{fname}`, "
    "no LapTuple should appear in keyword arguments.")


#### Tree based utility functions ####
# Below are several functions that work on args and kwargs.
# The major role is to add / remove / count / check / replace LapTuple.

def lap_setter (x, set_to: Callable):
  if type(x) in [list, tuple, dict]:
    return iter_func(x, lap_setter, (set_to, ))
  return set_to(x)

def lap_checker (x, unused = None):
  """Check whether an element is laptuple.
  If x is Python Container, iterate and recall lap_checker on its element.
  """
  if type(x) in [list, tuple, dict]:
    return iter_func(x, lap_checker, (unused, ))
  return x.spars if isinstance(x, LapTuple) else False

def lap_counter (x, unused = None):
  """Count the total number of laptuple in x.
  """
  if type(x) in [list, tuple]:
    return sum([lap_counter(w, (unused, )) for w in x])
  elif type(x) == dict:
    return sum([lap_counter(w, (unused, )) for w in x.values()])
  else:
    return int(isinstance(x, LapTuple))

def lap_axeser (x, mode, idx = 0):
  """Map laptuple to 0, and the rest to None. Used for vmap.
  """
  if type(x) in [list, tuple, dict]:
    return iter_func(x, lap_axeser, (mode, idx))
  if mode == "in":
    return idx if isinstance(x, LapTuple) else None
  elif mode == "out":
    return idx if isinstance(x, jnp.ndarray) else None

def lap_extractor (x, idx: TupType):
  """Get specific value in laptuple, and leave other unchanged.
  """
  if type(x) in [list, tuple, dict]:
    return iter_func(x, lap_extractor, (idx, ))
  return x.get(idx) if isinstance(x, LapTuple) else x

def tupler (x, unused = None):
  """Transfer all laptuple to normal tuple
  """
  if type(x) in [list, tuple, dict]:
    return iter_func(x, tupler, (unused, ))
  return x.to_tuple() if isinstance(x, LapTuple) else x

def laptupler (iterable, indicator,
               trans = None,
               ):
  """Transfer all tuple where indicator is True to something based on trans.
  Default is to laptuple. 
  
  """
  typ = type(indicator)
  if typ == list:
    return [laptupler(*w, trans) for w in zip(iterable, indicator)]
  elif typ == tuple:
    return tuple([laptupler(*w, trans) for w in zip(iterable, indicator)])
  elif typ == dict:
    return {k:laptupler(iterable[k], v, trans) for k,v in indicator.items()}
  if trans is not None:
    return trans(iterable, indicator)
  # By default, indicator instance is its sparsity
  return LapTuple(*iterable, indicator) if indicator else iterable


def iter_func (iterable, op = lap_extractor, opargs = (TupType.VALUE, )):
  """Return a variable tree with the same structure as iterable.
  If an element is list, tuple or dict, call recursively;
  Otherwise, raise. 

  Args:
    iterable (Iterable): iterable should support method `iter`.
    op (function): a function that applies to each element in iterable.
    args : parameters that pass to op function. 
  """
  iter_type = type(iterable)

  if iter_type == list:
    return [op(x, *opargs) for x in iterable]
  elif iter_type == tuple:
    return tuple([op(x, *opargs) for x in iterable])
  elif iter_type == dict:
    return {k: op(v, *opargs) for k,v in iterable.items()}
  raise TypeError(f"Iterable should has type `dict`, `list`, or `tuple`, but got {iter_type}.")

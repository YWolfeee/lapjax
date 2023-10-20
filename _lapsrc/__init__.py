from lapjax.lapsrc.laptuple import (
  LapTuple as LapTuple, 
  TupType as TupType,
)
from lapjax.lapsrc.lapconfig import lapconfig as lapconfig
from lapjax.lapsrc.functions import (
  vmap as vmap,
  FType as FType,
  custom_wrap as custom_wrap,
  is_wrapped as is_wrapped,
  get_wrap_by_f as get_wrap_by_f,
)
from lapjax.lapsrc.sparsinfo import (
  InputInfo as InputInfo,
  SparsInfo as SparsInfo,
)

import os as _os
rig_files = [w for w in _os.listdir(__path__[0]) if 
             w.endswith('.py') and w not in ['__init__.py', 'config.py']]
for w in rig_files:
    try:
      exec(f'from lapjax import {w[:-3]} as {w[:-3]}')
    except Exception as e:
      pass
      # print(f"Lapjax Warning: when wrapping '{w}',",
      #       f"got ImportError:\n    {e}\n" + \
      #       "It won't affect unless you manually import this moudle.")
del rig_files
del _os

### Over-write module entrance to lapjax ###
from lapjax import _src as _src
from lapjax import abstract_arrays as abstract_arrays
from lapjax import api_util as api_util
from lapjax import distributed as distributed
from lapjax import debug as debug
from lapjax import dtypes as dtypes
from lapjax import errors as errors
from lapjax import image as image
from lapjax import lax as lax
from lapjax import nn as nn
from lapjax import numpy as numpy
from lapjax import ops as ops
from lapjax import profiler as profiler
from lapjax import random as random
from lapjax import stages as stages
from lapjax import tree_util as tree_util
from lapjax import util as util

import sys, importlib
from lapjax.lapsrc.wrapper import _wrap_module
_wrap_module(importlib.import_module(__name__.replace('lapjax', 'jax')), 
             sys.modules[__name__])
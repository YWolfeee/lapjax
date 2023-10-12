from lapjax.laptuple import (
  LapTuple as LapTuple, 
  TupType as TupType,
)
from lapjax.lapconfig import lapconfig as lapconfig
from lapjax.functions import (
  vmap as vmap,
  FType as FType,
  custom_wrap as custom_wrap
)
from lapjax.sparsinfo import (
  InputInfo as InputInfo,
  SparsInfo as SparsInfo,
)

### Over-write module entrance to lapjax ###
# from lapjax import numpy as numpy
# from lapjax import nn as nn
# from lapjax import lax as lax 
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
from lapjax.wrapper import _wrap_module
_wrap_module(importlib.import_module(__name__.replace('lapjax', 'jax')), 
             sys.modules[__name__])
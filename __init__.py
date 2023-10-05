from lapjax.laptuple import (
  LapTuple as LapTuple, 
  TupType as TupType,
)
from lapjax.lapconfig import lapconfig as lapconfig
from lapjax.functions import (
  vmap as vmap,
  FType as FType,
)
from lapjax.sparsinfo import (
  InputInfo as InputInfo,
  SparsInfo as SparsInfo,
)
from lapjax.wrapper import custom_wrap as custom_wrap

### Over-write module entrance to lapjax ###
from lapjax import numpy as numpy
from lapjax import nn as nn
from lapjax import lax as lax 

import sys, importlib
from lapjax.wrapper import _wrap_module
_wrap_module(importlib.import_module(__name__.replace('lapjax', 'jax')), 
             sys.modules[__name__])
from jax import *
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
import lapjax.numpy as numpy
import lapjax.nn as nn
import lapjax.lax as lax 

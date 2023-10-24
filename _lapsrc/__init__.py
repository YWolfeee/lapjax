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
from lapjax.lapsrc.check_utils import create_check_function

### Over-write module entrance to lapjax ###
import os as _os
ignored = ['__pycache__']
submodules = [w for w in _os.listdir(__path__[0]) if 
              _os.path.isdir(__path__[0]+'/'+w) and w not in ignored
              or w.endswith('.py') and w not in ['__init__.py', 'config.py']]
for w in submodules:
    name = w[:-3] if w.endswith('.py') else w
    try:
      exec(f'from lapjax import {name} as {name}')
    except Exception as e:
      lapconfig.logger.warning(f"When wrapping `jax` modules `{w}`, " +
                               f"got ImportError:\n    {e}")
      lapconfig.logger.warning("This won't affect functions of other modules.")
del submodules
del _os

import sys, importlib
from lapjax.lapsrc.wrapper import _wrap_module
_wrap_module(importlib.import_module(__name__.replace('lapjax', 'jax')), 
             sys.modules[__name__])
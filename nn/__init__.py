import sys, importlib
from lapjax.wrapper import _wrap_module
_wrap_module(importlib.import_module(__name__.replace('lapjax', 'jax')), 
             sys.modules[__name__])
from jax.lax import *
from lapjax.wrapper import lapwrapper as wraps

stop_gradient = wraps(stop_gradient)
rsqrt = wraps(rsqrt)
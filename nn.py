from jax.nn import *
from lapjax.wrapper import lapwrapper as wraps

logsumexp = wraps(logsumexp)
softmax = wraps(softmax)
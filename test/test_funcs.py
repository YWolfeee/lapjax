import time

import jax
from jax.config import config as jax_config

from lapjax import FType, create_check_function
import lapjax.numpy as jnp
from lapjax.lapsrc.wrap_list import wrap_func_dict
from lapjax.lapsrc.wrapper import _lapwrapper
jax_config.update("jax_enable_x64", True)

print("Start checking wrapped functions...")

def check_diff(func, *x, derivative_inputs=0, derivative_outputs=0, **kw):
  lapfunc = _lapwrapper(func)
  check_function = create_check_function(
    lapfunc, derivative_inputs, derivative_outputs, seed=int(time.time())
  )

  grad_diff, lap_diff = check_function(
    *x,
    **kw,
  )
  print(
    f"Function[{func.__name__}] difference of gradient: {grad_diff:.2e}, difference of Laplacian: {lap_diff:.2e}"
  )
  try:
    assert (
      grad_diff < 1e-8 and lap_diff < 1e-8
    ), f"Abnormal difference in :{func}. Gradient difference and Laplacian difference should be smaller than 1e-8."
  except AssertionError as e:
    logger.error(e)
    raise e

def test_all_CONSTRUCTION():
  print("Skip CONSTRUCTION functions")

def test_all_LINEAR():
  print("Test LINEAR functions...")

  check_diff(jax.numpy.reshape, jnp.ones([12, ]), (3, 4))
  check_diff(jax.numpy.transpose, jnp.ones([6, 8]))
  check_diff(jax.numpy.swapaxes, jnp.ones([6, 8, 10]), 1, 2)
  check_diff(jax.numpy.split, jnp.ones([6, 8]), 2)
  check_diff(jax.numpy.array_split, jnp.ones([6, 8]), 2)
  check_diff(jax.numpy.concatenate, [jnp.ones([3, 2]), jnp.ones([3, 4])], axis=-1)
  check_diff(jax.numpy.squeeze, jnp.ones([3, 2, 1]))
  check_diff(jax.numpy.expand_dims, jnp.ones([3, 2]), 1)
  check_diff(jax.numpy.repeat, jnp.ones([3, 2]), repeats=4)
  check_diff(jax.numpy.tile, jnp.ones([3, 4]), reps=2)
  check_diff(
    jax.numpy.where,
    jnp.ones([3, 4]),
    jnp.ones([3, 4]) * 2,
    jnp.ones([3, 4]) * 3,
    derivative_inputs=(1, 2),
  )
  check_diff(jax.numpy.triu, jnp.ones([3, 4]))
  check_diff(jax.numpy.tril, jnp.ones([3, 4]))
  check_diff(jax.numpy.sum, jnp.ones([3, 4]))
  check_diff(jax.numpy.mean, jnp.ones([3, 4]))
  check_diff(jax.numpy.broadcast_to, jnp.ones([3, 4]), (5, 3, 4))

  print("LINEAR functions checked")

def test_all_ELEMENT():
  print("Test ELEMENT functions...")

  for func in wrap_func_dict[FType.ELEMENT]:
    if func in [jax.numpy.power, jax.lax.pow]:
      check_diff(func, jnp.ones([3, 4]) * 0.1, 2.0)
    elif func == jax.lax.integer_pow:
      check_diff(func, jnp.ones([3, 4]) * 0.1, 2)
    elif func == jax.lax.acosh or func == jax.numpy.arccosh:
      check_diff(func, jnp.ones([3, 4]) * 10)
    else:
      check_diff(func, jnp.ones([3, 4]) * 0.1)

  print("ELEMENT functions checked")

def test_all_OVERLOAD():
  print("Skip OVERLOAD functions")

def test_all_MERGING():
  print("Test MERGING functions...")

  check_diff(jax.numpy.linalg.norm, jnp.ones([3, 4]))
  check_diff(jax.numpy.prod, jnp.ones([3, 4]))

  print("MERGING functions checked")

def test_all_CUSTOMIZED():
  print("Test CUSTOMIZED functions...")

  check_diff(
    jax.numpy.matmul, jnp.ones([3, 4]), jnp.ones([4, 5]), derivative_inputs=(0, 1)
  )
  check_diff(
    jax.numpy.dot, jnp.ones([3, 4]), jnp.ones([4, 5]), derivative_inputs=(0, 1)
  )
  check_diff(jax.numpy.max, jnp.ones([3, 4]))
  check_diff(jax.numpy.min, jnp.ones([3, 4]))
  check_diff(jax.numpy.amax, jnp.ones([3, 4]))
  check_diff(jax.numpy.amin, jnp.ones([3, 4]))
  check_diff(
    jax.numpy.linalg.slogdet, jnp.array([[1, 2.0], [2, 1]]), derivative_outputs=1
  )
  check_diff(jax.nn.logsumexp, jnp.array([[1, 2], [3, 4.0]]), axis=-1)
  check_diff(jax.nn.softmax, jnp.array([[1, 2], [3, 4.0]]), axis=-1)

  print("CUSTOMIZED functions checked")

test_all_CONSTRUCTION()
test_all_LINEAR()
test_all_ELEMENT()
test_all_OVERLOAD()
test_all_MERGING()
test_all_CUSTOMIZED()
print("All functions are checked. Test passed.")

import jax
from jax import numpy as jnp
from jax import lax as jlax

from lapjax.function_class import FType

wrap_func_dict = {
  FType.CONSTRUCTION: [
    jnp.shape, jnp.size,
    jnp.eye, jnp.array,
    jnp.ones, jnp.ones_like,
    jnp.zeros, jnp.zeros_like,
    jnp.asarray, jnp.sign,
    jlax.stop_gradient,
  ],

  FType.LINEAR: [
  jnp.reshape, jnp.transpose, jnp.swapaxes,
  jnp.split, jnp.array_split, jnp.concatenate,
  jnp.squeeze, jnp.expand_dims,
  jnp.repeat, jnp.tile,
  jnp.where, jnp.triu, jnp.tril,
  jnp.sum, jnp.mean,
  jnp.broadcast_to,
  ],

  FType.ELEMENT: [
  jnp.sin, jnp.cos, jnp.tan,
  jnp.arcsin, jnp.arccos, jnp.arctan,
  jnp.arcsinh, jnp.arccosh, jnp.arctanh,
  jnp.sinh, jnp.cosh, jnp.tanh,
  jnp.exp, jnp.log, jnp.exp2, jnp.log2,
  jnp.square, jnp.sqrt, jnp.power,
  jnp.abs, jnp.absolute,
  jlax.sin, jlax.cos, jlax.tan,
  jlax.asin, jlax.acos, jlax.atan, jlax.atan2,
  jlax.asinh, jlax.acosh, jlax.atanh, 
  jlax.exp, jlax.log,
  jlax.square, jlax.sqrt, jlax.rsqrt, 
  jlax.pow, jlax.integer_pow,
  jlax.abs, 
  ],

  FType.OVERLOAD: [
    jnp.add, jnp.subtract, jnp.multiply, jnp.divide, jnp.true_divide
  ],

  FType.MERGING: [
    jnp.linalg.norm, jnp.prod,
  ],

  FType.CUSTOMIZED: [
    jnp.matmul, jnp.dot,
    jnp.max, jnp.min,
    jnp.amax, jnp.amin,
    jnp.linalg.slogdet,
    jax.nn.logsumexp,
    jax.nn.softmax,
  ],

  FType.EMPTY: [
    jax.vmap,
  ],

}
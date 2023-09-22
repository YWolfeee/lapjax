# This module is a partially wrapped jax.numpy.
# For all functions wrapped below), wraps(we supported their computation
# for LapTuple.
# 
# To add a new function:
# 1.    Specify its behavior by modifying function_class), wraps(e.g.
#       add to an existing function class), wraps(or customized it.
# 2.    Wrap the function below. 
# 

from jax.numpy import *
from lapjax.wrapper import lapwrapper as wraps


#### FLinear class ####
reshape = wraps(reshape)
transpose = wraps(transpose)
swapaxes = wraps(swapaxes)
split = wraps(split)
array_split = wraps(array_split)
concatenate = wraps(concatenate)
squeeze = wraps(squeeze)
expand_dims = wraps(expand_dims)
repeat = wraps(repeat)
tile = wraps(tile)
# where = wraps(where) Not allow
triu = wraps(triu)
tril = wraps(tril)
array = wraps(array)

sum = wraps(sum)
mean = wraps(mean)
# einsum = wraps(einsum)

#### Construction Class ####
eye = wraps(eye)
ones = wraps(ones)
ones_like = wraps(ones_like)
zeros = wraps(zeros)
zeros_like = wraps(zeros_like)
asarray = wraps(asarray)
sign = wraps(sign)


#### FElement class ####
sin = wraps(sin)
cos = wraps(cos)
tan = wraps(tan)
arcsin = wraps(arcsin)
arccos = wraps(arccos)
arctan = wraps(arctan)
arcsinh = wraps(arcsinh)
arccosh = wraps(arccosh)
arctanh = wraps(arctanh)
sinh = wraps(sinh)
cosh = wraps(cosh)
tanh = wraps(tanh)
exp = wraps(exp)
log = wraps(log)
square = wraps(square)
sqrt = wraps(sqrt)
power = wraps(power)
abs = wraps(abs)
absolute = wraps(absolute)

#### Overload Class ####
add = wraps(add)
subtract = wraps(subtract)
multiply = wraps(multiply)
divide = wraps(divide)
true_divide = wraps(true_divide)

#### Merging Class ####
linalg.norm = wraps(linalg.norm)
prod = wraps(prod)

#### Customized Class ####
matmul = wraps(matmul)
dot = wraps(dot)
max = wraps(max)
min = wraps(min)
amax = wraps(amax)
amin = wraps(amin)
linalg.slogdet = wraps(linalg.slogdet)
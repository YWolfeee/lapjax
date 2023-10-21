import logging
import time

import jax

import lapjax
import lapjax.numpy as jnp

from lapjax.lapsrc.wrap_list import wrap_func_dict
from lapjax.lapsrc.wrapper import _lapwrapper
from lapjax.lapsrc.function_class import FType

from test_utils import create_check_function

logging.basicConfig(level=logging.INFO)

def check_diff(func, *x,derivative_inputs=0,derivative_outputs=0,**kw):
    lapfunc = _lapwrapper(func)
    check_function = create_check_function(lapfunc,derivative_inputs,derivative_outputs,seed=int(time.time()))
    
    grad_diff, lap_diff = check_function(*x,**kw,)
    logging.info(f'{func} gradient difference: {grad_diff}, Laplacian difference: {lap_diff}')
    if grad_diff > 1e-4 or lap_diff > 1e-4:
        logging.warning('Unnormal difference. Make sure you have enclosed the TF32 options before import JAX')

if __name__ == '__main__':
    logging.info('Start checking wrapped functions')
    logging.info('Skip EMPTY and CONSTRUCTION function')
    logging.info('LINEAR starts:')
    check_diff(jax.numpy.reshape,jnp.ones([12,]),(3,4))
    check_diff(jax.numpy.transpose,jnp.ones([6,8]))
    check_diff(jax.numpy.swapaxes,jnp.ones([6,8,10]),1,2)
    check_diff(jax.numpy.split,jnp.ones([6,8]),2)
    check_diff(jax.numpy.array_split,jnp.ones([6,8]),2)
    check_diff(jax.numpy.concatenate,[jnp.ones([3,2]),jnp.ones([3,4])],axis=-1)
    check_diff(jax.numpy.squeeze,jnp.ones([3,2,1]))
    check_diff(jax.numpy.expand_dims, jnp.ones([3,2]),1)
    check_diff(jax.numpy.repeat, jnp.ones([3,2]),repeats=4)
    check_diff(jax.numpy.tile, jnp.ones([3,4]),reps=2)
    check_diff(jax.numpy.where, jnp.ones([3,4]),jnp.ones([3,4])*2, jnp.ones([3,4])*3, derivative_inputs=(1,2))
    check_diff(jax.numpy.triu, jnp.ones([3,4]))
    check_diff(jax.numpy.tril, jnp.ones([3,4]))
    check_diff(jax.numpy.sum, jnp.ones([3,4]))
    check_diff(jax.numpy.mean, jnp.ones([3,4]))
    check_diff(jax.numpy.broadcast_to,jnp.ones([3,4]),(5,3,4))

    logging.info('LINEAR checked. ELEMENT starts:')

    for func in wrap_func_dict[FType.ELEMENT]:
        if func == jax.numpy.power or func == jax.lax.pow or func == jax.lax.integer_pow:
            check_diff(func, jnp.ones([3,4])*0.1, 2)
        elif func == jax.lax.atan2:
            check_diff(func, jnp.ones([3,4])*0.1, jnp.ones([3,4])*0.1, derivative_inputs=(0,1))
        elif func == jax.lax.acosh or func == jax.numpy.arccosh:
            check_diff(func, jnp.ones([3,4])*10)
        else:
            check_diff(func, jnp.ones([3,4])*0.1)
    
    logging.info('ELEMENT checked. MERGING starts:')
    check_diff(jax.numpy.linalg.norm, jnp.ones([3,4]))
    check_diff(jax.numpy.prod, jnp.ones([3,4]))

    logging.info('MERGING checked. OVERLOAD skipped. CUSTOMIZED starts:')
    
    check_diff(jax.numpy.matmul, jnp.ones([3,4]),jnp.ones([4,5]), derivative_inputs=(0,1))
    check_diff(jax.numpy.dot, jnp.ones([3,4]),jnp.ones([4,5]), derivative_inputs=(0,1))
    check_diff(jax.numpy.max, jnp.ones([3,4]))
    check_diff(jax.numpy.min, jnp.ones([3,4]))
    check_diff(jax.numpy.amax, jnp.ones([3,4]))
    check_diff(jax.numpy.amin, jnp.ones([3,4]))
    check_diff(jax.numpy.linalg.slogdet, jnp.array([[1,2.0],[2,1]]),derivative_outputs=1)
    check_diff(jax.nn.logsumexp, jnp.array([[1,2],[3,4.0]]),axis=-1)
    check_diff(jax.nn.softmax, jnp.array([[1,2],[3,4.0]]),axis=-1)



    



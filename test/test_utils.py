import itertools
import logging

import jax

import lapjax
import lapjax.numpy as jnp

from lapjax import LapTuple
logging.basicConfig(level=logging.INFO)


def _prod(x):
    res=1
    for t in x:
        res *= t
    return res

def _convert_pytree_to_vector(pytree):
    list_of_array, tree_structure = jax.tree_util.tree_flatten(pytree)
    shape_list = [a.shape for a in list_of_array]
    vector = lapjax.numpy.concatenate([a.reshape(-1) for a in list_of_array])
    return vector, shape_list, tree_structure

def _convert_vector_to_pytree(vector, shape_list, tree_structure):
    len_list = [_prod(shape) for shape in shape_list]
    list_of_array = lapjax.numpy.split(vector, list(itertools.accumulate(len_list))[:-1])
    list_of_array = [a.reshape(shape) for a,shape in zip(list_of_array, shape_list)]
    pytree = jax.tree_util.tree_unflatten(tree_structure, list_of_array)
    return pytree

def _convert_grad_pytree_to_vector(grad_pytree):
    list_of_array, tree_structure = jax.tree_util.tree_flatten(grad_pytree)
    input_dim = list_of_array[0].shape[0]
    shape_list = [a.shape[1:] for a in list_of_array]
    grad_vector = lapjax.numpy.concatenate([a.reshape((input_dim,-1)) for a in list_of_array])
    return grad_vector, shape_list, tree_structure

def _convert_grad_vector_to_pytree(grad_vector, shape_list, tree_structure):
    input_dim = grad_vector.shape[0]
    len_list = [_prod(shape) for shape in shape_list]
    list_of_array = lapjax.numpy.split(grad_vector, list(itertools.accumulate(len_list))[:-1],axis=-1)
    list_of_array = [a.reshape((input_dim,)+shape) for a,shape in zip(list_of_array, shape_list)]
    pytree = jax.tree_util.tree_unflatten(tree_structure, list_of_array)
    return pytree

def create_check_function(test_func, derivative_args=0, derivative_outputs=0,  input_dim=3, seed=1234, return_all=False):
    '''
    Create a function used for testing the Laplacian calculation.
    Inputs:
        derivative_args: An int or a tuple of ints, denotes the argnum used in the derivative calcalution, i.e., 
            should be replaced by LapTuple in LapJAX.
        derivative_output: Denotes which arguments should be considered as a LapTuple in the output. 
        input_dim: The testing input dimension in this function.
        seed: random seed
        return_all: If this term is setted as True, the checking function will return grad_in_hessian, 
            grad_in_lapjax, lap_in_hessian, lap_in_lapjax; As for the False, it will return the difference
            between hessian and lapjax methods.
    Outputs:
        checking_function. The input of this function should be same as the original function. It outputs 
            the difference between LAPJAX and Hessian method in calculation the Laplacian and Gradient.
            WARNING: the arguments whose position is recorded in derivative_args, should be POSITIONAL ARGUMENT. 
            Put them as keyword arguments, e.g. x=ABC, would lead to unexpected behavior.
    '''
    
    if isinstance(derivative_args, int):
        derivative_args = (derivative_args, )
    if isinstance(derivative_outputs, int):
        derivative_outputs = (derivative_outputs, )


    def checking_function(*args, **kwargs):
        # This function in fact used a MLP to generate the input for test_func
        # It thus contains a non-zero gradient and laplacian term.


        # create a test_function only takes the tracking args as input and output
        def minimal_test_func(*dargs):
            tmpargs = []
            tpi=0
            for pl, targ in enumerate(args):
                if pl in derivative_args:
                    tmpargs.append(dargs[tpi])
                    tpi = tpi + 1
                else:
                    tmpargs.append(args[pl])
            tmpargs = tuple(tmpargs)
            outputs = test_func(*tmpargs,**kwargs)
            if not isinstance(outputs, tuple):
                outputs = outputs,
            minimal_outputs = []
            for pl in derivative_outputs:
                minimal_outputs.append(outputs[pl])
            return tuple(minimal_outputs)

        # counting the input and output vector shape
        tracking_args = [args[pl] for pl in derivative_args]
        tracking_vector, shape_list, tree_structure = _convert_pytree_to_vector(tracking_args)
        output = test_func(*args,**kwargs)
        if not isinstance(output, tuple):
                output = output,
        tracking_output = [output[pl] for pl in derivative_outputs]
        tracking_output_vector, output_shape_list, output_tree_structure = _convert_pytree_to_vector(tracking_output)

        # generate MLP parameter
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        w1 = jax.random.normal(subkey, [256,input_dim])
        key, subkey = jax.random.split(key)
        w2 = jax.random.normal(subkey, [len(tracking_vector), 256]) / 16 
        # make the output of MLP is the same as tracking vector
        b2 = tracking_vector - jnp.tanh(jnp.matmul(w2, jnp.tanh(jnp.dot(w1, jax.numpy.ones([input_dim])))))

        def MLP(x):
            return jnp.tanh(jnp.matmul(w2, jnp.tanh(jnp.dot(w1, x)))) + b2
        
        key, subkey = jax.random.split(key)
        w3 = jax.random.normal(subkey, [len(tracking_output_vector),])
        
        # create a scalar function for checking the Laplacian of the final output
        def scalar_function(x):
            x = MLP(x)
            inputs = _convert_vector_to_pytree(x, shape_list, tree_structure)
            outputs = minimal_test_func(*inputs)
            vector_output,_,_ = _convert_pytree_to_vector(outputs)
            return jnp.sum(w3*vector_output)
        
        # result of LapJAX
        lp_input = LapTuple(jnp.ones([input_dim,]),is_input=True)
        # we only check the dense setting here. The sparsity supports should be checked case by case.
        lp_input = lp_input.set_dense(True) 
        lp_output = scalar_function(lp_input)
        
        # result of Hessian
        def hessian(x):
            grad = jax.grad(scalar_function)(x)
            lap = jnp.trace(jax.hessian(scalar_function)(x))
            return grad, lap
        grad, lap = hessian(jnp.ones([input_dim,]))
        if return_all:
            return grad, lp_output.grad, lap, lp_output.lap
        else:
            return jnp.linalg.norm(grad-lp_output.grad), jnp.linalg.norm(lap-lp_output.lap)
    
    return checking_function

if __name__ == '__main__':
    
    check_func = create_check_function(lapjax.numpy.sin)
    grad_diff, lap_diff = check_func(jnp.ones([6,3]))
    logging.info(f'gradient difference: {grad_diff}, Laplacian difference: {lap_diff}')
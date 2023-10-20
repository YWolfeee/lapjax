from inspect import isclass
from functools import wraps
from lapjax.lapsrc.functions import F, FType, lap_dispatcher, is_wrapped

def _lapwrapper (wrapped_f: F) -> F:
  """Lapjax wrapper functions. This returns the entrance of wrapper.
  If first checks whether LapTuple exists in args and kwargs.
  If not, return standard call on wrapped_f.
  Otherwise, dispatch to `lap_dispatcher`, which will process input 
    according to wrapped_f.
  """
  from lapjax.lapsrc.laputils import iter_func, lap_counter, lap_checker, tupler
  
  @wraps(wrapped_f)
  def entrance (*args, **kwargs):
    lap_num = lap_counter(args) + lap_counter(kwargs)
    if lap_num == 0: # No laptuple iterupts. Directly return.
      return wrapped_f(*args, **kwargs)

    if not is_wrapped(wrapped_f):
      raise NotImplementedError(
        f"Lapjax encounters unwrapped function '{wrapped_f.__name__}'.\n" + \
         "Please consider using other functions or wrap it yourself.\n" + \
         "You can refer to README for more information about customized wrap."
        )

    # Pass wrapped_f and LapTuple indicator to dispatcher.
    tuplized_f = lap_dispatcher(wrapped_f, 
                                lap_checker(args),
                                lap_checker(kwargs))

    # Compute the value with tuplized arguments.
    return tuplized_f(*iter_func(args, tupler), 
                      **iter_func(kwargs, tupler))
  
  entrance.__hash__ = wrapped_f.__hash__  # used as main key in wrap classes
  return entrance

def _wrap_module (module, new_module):
  alls = [w for w in dir(module) if not w.startswith('_')]
  for name in alls:
    if hasattr(new_module, name):
      continue
    val = getattr(module, name)
    if callable(val) and not isclass(val):
      setattr(new_module, name, _lapwrapper(val))
    else:
      setattr(new_module, name, val)
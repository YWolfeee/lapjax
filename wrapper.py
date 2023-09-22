from lapjax.func_utils import wraps
from lapjax.laputils import iter_func, lap_counter, lap_checker, tupler
from lapjax.functions import F, FType, lap_dispatcher

def lapwrapper (wrapped_f: F, 
                custom_type: FType = None) -> F:
  """Lapjax wrapper functions. This returns the entrance of wrapper.
  If first checks whether LapTuple exists in args and kwargs.
  If not, return standard call on wrapped_f.
  Otherwise, dispatch to `lap_dispatcher`, which will process input 
    according to wrapped_f.
  """

  docstr = ("Lapjax wrapped '{fun}' function."
            "We support its call with LapTuple inputs.")
  if wrapped_f.__doc__:
    docstr += "\n\nOriginal documentation:\n\n" + wrapped_f.__doc__

  @wraps(wrapped_f, docstr=docstr)
  def entrance (*args, **kwargs):
    lap_num = lap_counter(args) + lap_counter(kwargs)
    # print(f"---- Entering wrappers, calling {wrapped_f.__name__}. #Lap = {lap_num} ----")
    if lap_num == 0: # No laptuple iterupts. Directly return.
      return wrapped_f(*args, **kwargs)

    # Pass wrapped_f and LapTuple indicator to dispatcher.
    tuplized_f = lap_dispatcher(wrapped_f, 
                                custom_type,
                                lap_checker(args),
                                lap_checker(kwargs))

    # Compute the value with tuplized arguments.
    return tuplized_f(*iter_func(args, tupler), 
                      **iter_func(kwargs, tupler))
  
  return entrance

def custom_wrap(f: F, custom_type: FType):
  """Bind a self-defined function f to a funtion type.
  This will allow the dispatch to treat the return function as this type.

  Args:
      f (F): the function you want to bind to a predefined type.
      custom_type (FType): the function type you want to bind to.

  Returns:
      lapjax wrapped f.
  """
  print(f"Customing function '{f.__name__}' as {custom_type}.")
  return lapwrapper(f, custom_type)

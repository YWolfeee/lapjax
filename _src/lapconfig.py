import enum


class AutoLap(object):
    """Specify the auto laplacian mode for general `Merging` function.
      `hessian` means we use nabla.T @ h @ nabla to compute.
      `fori_loop` means we use jax.linearize to compute.
      When `fori_loop` is used, `block` specifies how many rows are computed at once.
    """
    class Type(enum.Enum):
      hessian = enum.auto()
      fori_loop = enum.auto()

    def __init__(self) -> None:
      self.type = self.Type.hessian
      self.block = 16

class LapConfig(object):
  """Lapjax configuration.
  
  You can import lapconfig by `from lapjax import lapconfig` and change settings.
  
  """
  def __init__(self) -> None:
    self.autolap = AutoLap()    
    self.debug_print = False
    
lapconfig = LapConfig()

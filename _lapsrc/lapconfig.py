import enum
import logging


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
    self.logger = logging.getLogger("LapJAX")
    logging.basicConfig(format='%(levelname)s: %(name)s:  %(message)s')
    self.autolap = AutoLap()
    self.linear_wrap_warned = False
    self.merging_wrap_warned = False

  def log(self, msg: str, level=logging.INFO):
    """Log a message using logging.Logger."""
    self.logger.log(level, msg)

  def set_autolap_hessian(self):
    "Set the merging function auto laplacian mode to `hessian`."
    self.autolap.type = AutoLap.Type.hessian

  def set_autolap_fori_loop(self, block: int = 16):
    """Set the merging function auto laplacian mode to `fori_loop`.
    `block` specifies how many rows are computed at once.
    Smaller `block` means less memory usage but less efficiency.
    """
    self.autolap.type = AutoLap.Type.fori_loop
    self.autolap.block = block
    
lapconfig = LapConfig()

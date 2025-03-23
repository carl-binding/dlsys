import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt( 6/(fan_in+fan_out))
    return rand( fan_in, fan_out, low=-a, high=+a)
    


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
  std = gain * math.sqrt( 2 / (fan_in + fan_out))
  ## there is a bug in the comments, std is *not* squared, sloppy...
  ## std_squared = std*std
  return randn( fan_in, fan_out, mean=0, std=std )

def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt( 2)
    bound = gain * math.sqrt( 3/fan_in)
    return rand( fan_in, fan_out, low=-bound, high=bound)


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt( 2)
    std = gain / math.sqrt( fan_in)
    return randn( fan_in, fan_out, mean=0, std=std)
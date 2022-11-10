
from jesse.helpers import get_candle_source
import numpy as np
# from numba import njit
# import talib 
from math import exp
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from talipp.indicators import EMA
import _pickle as cPickle
import gc
#jesse backtest  '2021-01-03' '2021-03-02'
'''
talipp test Alma, needs more work to become incremental, offset rounds down to nearest tenths place   
''' 
  
def t_ema(candles: np.ndarray, length:int=20, source_type: str = "close", sequential: bool = False, firstrun:bool=False) -> Union[float, np.ndarray]:    
    candles = candles[-480:] #slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)   
    # alma = ALMA(length, offset, sigma, source)
    global ema, ema_pickled 
    if firstrun == True:
        ema = EMA(length,source)
        ema.purge_oldest(len(source)-(length))
        ema_pickled = cPickle.dumps(ema,protocol=-1)
    else:
        gc.disable()
        preema = cPickle.loads(ema_pickled)
        preema.add_input_value(source[-1])
        preema.purge_oldest(1)
        ema = preema
        ema_pickled = cPickle.dumps(preema,protocol=-1)
        gc.enable()
    if sequential:
        return ema[-1]
    else:
        return ema[-1]
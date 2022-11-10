from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
import tulipy as ti
import math 

"""
https://www.tradingview.com/script/8YMTHXu3-TASC-2021-10-MAD-Moving-Average-Difference/#chart-view-comments
"""
#jesse backtest  '2021-01-03' '2021-03-02'

def movingaveragediff(candles: np.ndarray, longLength:int=23, shortLength:int=8,source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type) 
    shortAvg = talib.SMA(source,shortLength)
    longAvg = talib.SMA(source, longLength)
    mad = 100 * (shortAvg - longAvg) / longAvg
    if sequential:
        return mad
    else:
        return mad[-1]
        
    
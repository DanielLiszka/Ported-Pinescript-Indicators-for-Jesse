from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/a3OLA5yI-Ergodic-Mean-Deviation-Indicator-CC/#chart-view-comment-form

Ergodic Mean Deviation Indicator 
"""

def emdi(candles: np.ndarray, length1:int=32, length2:int=5,length3:int=5, source_type: str = "close", sequential: bool = False):
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    ma = talib.EMA(source,length1)
    diff = source - ma 
    diffMa = talib.EMA(diff,length2)
    edi = talib.EMA(diffMa, length3)
    
    if sequential:
        return edi
    else:
        return edi[-1]

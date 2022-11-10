from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/9bUUSzM3-Madrid-Trend-Squeeze/

Madrid Squeeze
"""

MTS = namedtuple("MTS",["closema","sqzma","refma"])

def mts(candles: np.ndarray, length:int=34,ref:int=13,sqzLen:int=5, source_type: str = "close", sequential: bool = False) -> MTS:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    ma = talib.EMA(source,length)
    closema = source - ma
    refma = talib.EMA(source,ref) - ma
    sqzma = talib.EMA(source,sqzLen)-ma 
    
    if sequential:
        return MTS(closema,sqzma,refma)
    else:
        return MTS(closema[-1],sqzma[-1],refma[-1])


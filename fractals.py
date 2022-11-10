from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
import tulipy as ti
import math 

Fractal = namedtuple('Fractals',['ftop','fbottem'])

#jesse backtest  '2021-01-03' '2021-03-02'

"""
https://www.tradingview.com/script/eUkreZBc-Fractals/#chart-view-comment-form
"""

def fractal(candles: np.ndarray,source_type: str = "close", sequential: bool = False) -> Fractal:
    candles = slice_candles(candles, sequential) if sequential else candles[-480:] 
    source = get_candle_source(candles, source_type=source_type)  
    fTop = 1 if (candles[:,3][-5] < candles[:,3][-3] and candles[:,3][-4] < candles[:,3][-3] and candles[:,3][-3] > candles[:,3][-2] and candles[:,3][-3] > candles[:,3][-1]) else 0
    fBottom = 1 if (candles[:,4][-5] > candles[:,4][-3] and candles[:,4][-4] > candles[:,4][-3] and candles[:,4][-3] < candles[:,4][-2] and candles[:,4][-3] < candles[:,4][-1]) else 0 
    if sequential: 
        return Fractal(fTop,fBottom)
    else:    
        return Fractal(fTop,fBottom)


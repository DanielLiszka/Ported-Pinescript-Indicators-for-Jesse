from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/IbqpNXSe-FIR-Hann-Window-Indicator-Ehlers/

FIR Hann Window Filter
"""

def hwf(candles: np.ndarray, source2:np.ndarray=None, length:int=20, source_type: str = "close", sequential: bool = False):
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    if source2:
        out = hwf_func(source,source2,length)
    else:
        out = hwf_func(source,source,length)
    
    if sequential:
        return out
    else:
        return out[-1]

@njit
def hwf_func(source1,source2,length):
    Deriv = np.zeros(source1.shape[0])
    out = np.zeros(source1.shape[0])
    Filt1 = np.zeros(source1.shape[0])
    for i in range(length+1,source1.shape[0]):
        Deriv[i] = source2[i] - source2[i-1] 
        Filt = 0.0
        coef = 0.0
        deriv2 = 0.0
        for count in range(1,length+1):
            Filt = Filt + (1 - np.cos(2 * np.pi * count / (length+1))) * Deriv[(i-(count-1))]
            coef = coef + (1 - np.cos(2 * np.pi * count / (length+1)))
        if coef != 0:
            Filt = Filt / coef
        Filt1[i] = Filt
        out[i] = source2[i] * (Filt1[i] - Filt1[i-1])
    return out
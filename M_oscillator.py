from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple 

M_OSCILLATOR = namedtuple('M_OSCILLATOR',['moLine', 'sigLine', 'linecolor'])

"""
https://www.tradingview.com/script/gxRRuSwr-M-Oscillator/
"""

def M_oscillator(candles: np.ndarray, length: int= 14, source_type: str = "close", sequential: bool = False) -> M_OSCILLATOR:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    M = func_moscillator(source, length)
    emaV = talib.EMA(M,5)
    moLine = talib.EMA(emaV,3)
    sigLine = talib.EMA(moLine,3)
    c_moLine = 1 if moLine[-1] > sigLine[-1] else -1     
    if sequential:
        return M_OSCILLATOR(moLine,sigLine,c_moLine)
    else:
        return M_OSCILLATOR(moLine[-1],sigLine[-1],c_moLine)


@njit(fastmath=True) 
def func_moscillator(source,length):
    final = np.zeros(source.shape[0])
    for i in range(source.shape[0]):
        t = 0.0 
        v = 0.0
        for j in range(1,length):
            t = 1 if source[i] > source[i-j] else -1 
            v = t + v 
        final[i] = v 
    return final 
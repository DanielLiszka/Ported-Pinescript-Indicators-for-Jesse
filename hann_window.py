from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

hannwindow = namedtuple('hannwindow',['hann', 'roc'])

def hann(candles: np.ndarray, length:int=20,source_type: str = "close", sequential: bool = False) -> hannwindow:
    candles = slice_candles(candles, sequential)
    source = candles[:,2] #get_candle_source(candles, source_type=source_type)
    hann,roc = func_hann(source,candles,length)
    if sequential:
        return hannwindow(hann, roc)
    else:
        return hannwindow(hann[-1], roc[-1])



@njit
def func_hann(source,candles,length):
    deriv = np.zeros(source.shape[0])
    filt1 = np.zeros(source.shape[0])
    roc = np.zeros(source.shape[0])
    for i in range(length+2,source.shape[0]):
        cosine = 0.0
        filt = 0.0
        coef = 0.0
        deriv[i] = (candles[:,2][i]) - (candles[:,1][i])
        for j in range(1,length+1):
            cosine = 1 - np.cos(360 * j / (length + 1) * np.pi / 180)
            filt = filt + (cosine * deriv[(i-j+1)])
            coef = coef + cosine 
        filt1[i] = filt/coef if coef != 0 else 0 
        roc[i] = (length / 6.28) * (filt1[i] - filt1[i-1])
    return filt1, roc
from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

EHRSI = namedtuple('EHRSI',['RSI','Signal'])

def ehrsi(candles: np.ndarray, length:int=14,source_type: str = "close", sequential: bool = False) -> EHRSI:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    hrsi,slo = func_ehrsi(source,candles,length)
    if slo[-1] > 0:
        if slo[-1] > slo[-2]:
            signal = 2 
        else: 
            signal = 1
    else:
        if slo[-1] < 0:
            if slo[-1] < slo[-2]:
                signal = -2
            else:
                signal = -1
        else:   
            signal = 0
    if sequential:
        return EHRSI(hrsi,signal)
    else:
        return EHRSI(hrsi[-1], signal)

@njit
def func_ehrsi(source,candles,length):
    upMa = np.zeros(source.shape[0])
    slo = np.zeros(source.shape[0])
    hrsi = np.zeros(source.shape[0])
    dnMa = np.zeros(source.shape[0])
    mom = np.zeros(source.shape[0])
    up = np.zeros(source.shape[0])
    down = np.zeros(source.shape[0])
    for i in range(source.shape[0]):
        mom[i] = source[i] - source[i-1] 
        up[i] = mom[i] if mom[i] > 0 else 0
        down[i] = np.abs(mom[i]) if mom[i] < 0 else 0 
        cosine = 0.0
        filt1 = 0.0
        coef = 0.0
        filt2 = 0.0
        for j in range(1,length+1):
            cosine = 1 - np.cos(2 * np.pi * j / (length + 1))
            filt1 += cosine * up[(i-j+1)]
            filt2 += cosine * down[(i-j+1)]
            coef += cosine 
        upMa[i] = filt1 / coef if coef != 0 else 0 
        dnMa[i] = filt2 / coef if coef != 0 else 0
        hrsi[i] = (upMa[i] - dnMa[i]) / (upMa[i] + dnMa[i]) * 100 if upMa[i] + dnMa[i] != 0 else 0 
        slo[i] = hrsi[i] - hrsi[i-1] 
        
    return hrsi,slo
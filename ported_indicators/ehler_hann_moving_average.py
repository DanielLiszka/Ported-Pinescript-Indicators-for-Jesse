from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

EHMA = namedtuple('EHMA',['Ma','Signal'])

def ehma(candles: np.ndarray, length:int=14,source_type: str = "close", sequential: bool = False) -> EHMA:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    filt,slo = func_ehma(source,candles,length)
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
        return EHMA(filt,signal)
    else:
        return EHMA(filt[-1], signal)

@njit
def func_ehma(source,candles,length):
    filt1 = np.zeros(source.shape[0])
    slo = np.zeros(source.shape[0])
    for i in range(source.shape[0]):
        cosine = 0.0
        filt = 0.0
        coef = 0.0
        for j in range(1,length+1):
            cosine = 1 - np.cos(2 * np.pi * j / (length + 1))
            filt += cosine * source[(i-j+1)]
            coef += cosine 
        filt1[i] = filt / coef if coef != 0 else 0 
        slo[i] = source[i] - filt1[i] 
    return filt1,slo
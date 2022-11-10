from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

#Inaccurate 

Median = namedtuple('Median',['MA','Signal'])

"""
https://www.tradingview.com/script/pq6FWb3Y-Ehlers-Median-Average-Adaptive-Filter-CC/#chart-view-comment-form
"""

def median(candles: np.ndarray, length:int=39,threshold:float=0.002,source_type: str = "close", sequential: bool = False) -> Median:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    MA,slo = func_median(source,length,threshold)
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
        return Median(MA,signal)
    else:
        return Median(MA,signal)

@jit(error_model="numpy")
def func_median(source,length,threshold):
    smth = np.zeros(source.shape[0])
    alpha1  = np.zeros(source.shape[0])
    filter1 = np.zeros(source.shape[0])
    slo = np.zeros(source.shape[0])
    for i in range(length+1,source.shape[0]):
        smth[i] = (source[i] + (2 * source[i-1]) + (2 * source[i-2]) + source[i-3])/6
        v3 = 0.2
        v2 = 0.0
        alpha = 0.0
        len1 = length
        while (v3 > threshold and len1 > 0):
            alpha = 2.0 / (len1+ 1)
            v1 = np.median(smth[i-(len1-1):i+1])
            #need to fix v2 previous multiplation 
            v2 = (alpha * smth[i]) + ((1 - alpha) * v2)
            v3 = np.abs(v1 - v2) / v1 if v1 != 0 else v3
            len1 -= 2
        len1 = 3 if len1 < 3 else len1
        alpha1[i] = 2 / (len1 + 1)
        filter1[i] = (alpha1[i] * smth[i]) + ((1 - alpha1[i]) * filter1[i-1])
        slo[i] = source[i] - filter1[i]
    return len1, slo
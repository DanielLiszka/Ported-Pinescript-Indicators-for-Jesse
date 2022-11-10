from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/XPZfVBWs-Detrended-Ehlers-Leading-Indicator-CC/#chart-view-comment-form

Detrended Ehlers Leading Indicator 
"""

DSP = namedtuple("DSP",["MA","Leading_MA"])

def dsp(candles: np.ndarray, length:int=14, source_type: str = "close", sequential: bool = False) -> DSP:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    dsp,deli = dsp_func(source,candles,length)
    if sequential:
        return DSP(dsp,deli)
    else:
        return DSP(dsp[-1],deli[-1])

@jit(error_model="numpy")
def dsp_func(source,candles,length):
    alpha = 2.0 / (length+1)
    price = np.zeros(source.shape[0])
    ma1 = np.zeros(source.shape[0])
    ma2 = np.zeros(source.shape[0])
    ma3 = np.zeros(source.shape[0])
    dsp = np.zeros(source.shape[0])
    deli = np.zeros(source.shape[0])
    for i in range(length,source.shape[0]):
        price[i] = (np.maximum(candles[:,3][i],candles[:,3][i-1]) + np.minimum(candles[:,4][i],candles[:,4][i-1]))/2
        ma1[i] = (alpha * price[i] ) + ((1 - alpha ) * ma1[i-1])
        ma2[i] = ((alpha/2)*price[i]) + ((1- (alpha /2))*ma2[i-1])
        dsp[i] = ma1[i] - ma2[i] 
        ma3[i] = (alpha * dsp[i]) + ((1 - alpha)* ma3[i-1])
        deli[i] = dsp[i] - ma3[i] 
    return dsp,deli
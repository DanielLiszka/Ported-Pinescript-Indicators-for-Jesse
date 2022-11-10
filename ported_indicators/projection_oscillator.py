from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib as ta
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/0jbAvmb0-Projection-Oscillator-CC/#chart-view-comment-form

Projection Oscillator
"""
PO = namedtuple("PO",["projection_oscillator","po_wma_signal"])

def po(candles: np.ndarray, length:int=14,smoothLength:int=4, source_type: str = "close", sequential: bool = False) -> PO:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    mHigh = linear_regression(candles[:,3],length,0) - linear_regression(candles[:,3],length,1)
    mLow = linear_regression(candles[:,4],length,0) - linear_regression(candles[:,4],length,1)
    po,signal = po_func(source,candles,length,smoothLength,mHigh,mLow)
    if sequential:
        return PO(po,signal)
    else:
        return PO(po[-1],signal[-1])

def linear_regression(series,length,offset):
    lri = ta.LINEARREG_INTERCEPT(series,length)
    lrs = ta.LINEARREG_SLOPE(series,length)
    return lri + lrs * (length - 1 - offset)
    
@njit
def po_func(source,candles,length,smoothLength,mHigh,mLow):
    res = np.zeros(source.shape[0])
    pbo = np.zeros(source.shape[0])
    for i in range(length,source.shape[0]):
        currH = 0.0
        prevH = 0.0
        currL = 0.0
        prevL = 0.0
        vHigh = 0.0
        vLow = 0.0
        upperBand = candles[:,3][i]
        lowerBand = candles[:,4][i]
        for j in range(length):
            currH = candles[:,3][i-j]
            prevH = candles[:,3][i-(j+1)]
            currL = candles[:,4][i-j]
            prevL = candles[:,4][i-(j+1)]
            vHigh = currH + (mHigh[i-j] * j)
            vLow = currL + (mLow[i-j] * j)
            upperBand = np.maximum(vHigh, upperBand)
            lowerBand = np.minimum(vLow, lowerBand)
        pbo[i] = 100 * (source[i] - lowerBand) / (upperBand - lowerBand) if upperBand - lowerBand != 0 else 0 
        weight = 0.0
        norm = 0.0 
        sum1 = 0.0
        for j in range(smoothLength):
            weight = (smoothLength - j)*smoothLength
            norm = norm + weight 
            sum1 = sum1 + pbo[i-j] * weight
        res[i] = sum1/norm 
    return pbo,res
    

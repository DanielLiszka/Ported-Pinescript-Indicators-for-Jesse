from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config
import jesse.indicators as ta

"""
https://www.tradingview.com/script/ks8m6BcG-Adaptive-Ehlers-Deviation-Scaled-Moving-Average-AEDSMA/

Adaptive Ehlers Deviation Scaled Moving Average (AEDSMA)

(Does not Appear to be accurate)
"""

def edsma(candles: np.ndarray, period:int=20,length:int=90, source_type: str = "close", sequential: bool = False):
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    edsma = edsma_func(source,candles,length,period)
    if sequential:
        return edsma
    else:
        return edsma[-1]


@njit
def edsma_func(source,candles,length,period):
    arg = np.sqrt(2) * np.pi / length
    a1 = np.exp(-arg)
    b1 = 2 * a1 * np.cos(arg)
    c3 = -np.power(a1,2)
    c1 = 1 - b1 - c3 
    newseries = np.zeros(source.shape[0])
    std1 = np.zeros(source.shape[0])
    alpha = np.zeros(source.shape[0])
    scaledFilter = np.zeros(source.shape[0])
    edsma = np.zeros(source.shape[0])
    avg = np.zeros(source.shape[0])
    avgZeros = np.zeros(source.shape[0])
    zeros = np.zeros(source.shape[0])
    for i in range(length, source.shape[0]):
        zeros[i] = source[i] - source[i-2]
        avgZeros[i] = (zeros[i] + zeros[i-1]) / 2 
        newseries[i] = c1 * avgZeros[i] + b1 * newseries[i-1] + c3 * newseries[i-2]
        sum1 = 0.0
        sum2 = 0.0
        avg[i] = np.mean(newseries[i-(length-1):i+1])
        for j in range(length):
            sum1 = (newseries[i-j] + -avg[i])
            sum2 = sum2 + sum1 * sum1 
        std1[i] = np.sqrt(sum2 / length)
        scaledFilter[i] = newseries[i] / std1[i] if std1[i] != 0 else 0 
        alpha[i] = 5 * np.abs(scaledFilter[i])/ length
        edsma[i] = alpha[i] * source[i] + (1 - alpha[i]) * edsma[i-1]
        
    return edsma
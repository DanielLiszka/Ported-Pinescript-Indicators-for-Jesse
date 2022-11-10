from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/EFw6gcWG-RS-Dynamic-Range-Commodity-Channel-Index-V0/

Dyanmic Range Commodity Channel Index
"""
DRCCI = namedtuple("DRCCI",["series","high_channel","low_channel"])


def drcci(candles: np.ndarray, length:int=20,smooth:int=1, source_type: str = "close", sequential: bool = False) -> DRCCI:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    talib_avg = talib.SMA(source,length)
    c,h,l = calt_func(source,length,smooth,talib_avg)
    if sequential:
        return DRCCI(c,h,l)
    else:
        return DRCCI(c[-1],h[-1],l[-1])

@jit(error_model="numpy")
def calt_func(source,length,smooth,talib_avg):
    altsource = np.zeros(source.shape[0])
    altsource_ma = np.zeros(source.shape[0])
    output = np.zeros(source.shape[0])
    sum1 = np.zeros(source.shape[0])
    sum2 = np.zeros(source.shape[0])
    max1 = np.zeros(source.shape[0])
    min1 = np.zeros(source.shape[0])
    h = np.zeros(source.shape[0])
    l = np.zeros(source.shape[0])
    pos = np.zeros(source.shape[0])
    neg = np.zeros(source.shape[0])
    sum2_avg = np.zeros(source.shape[0])
    alpha = 2 / (smooth+1)
    for i in range(length+1,source.shape[0]):
        altsource[i] = (source[i] - source[i-1])
        altsource_ma[i] = np.mean(altsource[i-(length-1):i+1])
        sum3 = 0.0
        sum4 = 0.0
        sum5 = 0.0
        for j in range(length):
            val = altsource[i-j]
            sum5 = sum5 + abs(val-altsource_ma[i])
        altsource_diff = sum5/length
        output[i] = (altsource[i] - altsource_ma[i]) / (0.015 * altsource_diff)
        sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = alpha * output[i] + (1 - alpha) * sum1[i-1]
        sum2[i-1] = 0 if np.isnan(sum2[i-1]) else sum2[i-1]
        sum2[i] = alpha * sum1[i] + (1 - alpha) * sum2[i-1]
        sum2_avg[i] = np.mean(sum2[i-(length-1):i+1])
        for j in range(length):
            sum3 = (sum2[i-j] + -(sum2_avg[i]))
            sum4 = sum4 + sum3 * sum3 
        pos[i] = np.sqrt(sum4 / length)
        neg[i] = -(pos[i])
        max1[i] = np.maximum(pos[i],neg[i])
        min1[i] = np.minimum(pos[i],neg[i])
        h[i] = np.amax(max1[i-(length-1):i+1])
        l[i] = np.amin(min1[i-(length-1):i+1])  
    return sum2, h , l 

from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/Dvwo6IPy-Relative-Strength-Volume-Adjusted-Exponential-Moving-Avg-CC/#chart-view-comment-form

Relative Strength Volume Adjusted Exponential Moving Average 
"""
RSVAEMA = namedtuple("RSVAEMA",["MA", "signal"])


def rsvaema(candles: np.ndarray, length:int=50,mult:float=10, source_type: str = "close", sequential: bool = False) -> RSVAEMA:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    rsvaEma, signal = fast_rsvaema(source,candles,length,mult)
    if sequential:
        return RSVAEMA(rsvaEma,signal)
    else:
        return RSVAEMA(rsvaEma[-1],signal[-1])

@njit
def fast_rsvaema(source,candles,length,mult):
    alpha = 2.0 / (length + 1)
    mom = np.zeros(source.shape[0])
    up = np.zeros(source.shape[0])
    down = np.zeros(source.shape[0])
    upMa = np.zeros(source.shape[0])
    dnMa = np.zeros(source.shape[0])
    rs = np.zeros(source.shape[0])
    rsvaEma = np.zeros(source.shape[0])
    slo = np.zeros(source.shape[0])
    signal = np.zeros(source.shape[0])
    
    for i in range(source.shape[0]):
        mom[i] = source[i] - source[i-1] 
        up[i] = candles[:,5][i] if mom[i] > 0 else 0 
        down[i] = candles[:,5][i] if mom[i] < 0 else 0
        upMa[i] = 0 if np.isnan(upMa[i-1]) else upMa[i-1]
        upMa[i] = alpha * up[i] + (1 - alpha) * upMa[i-1] 
        dnMa[i] = 0 if np.isnan(dnMa[i-1]) else dnMa[i-1] 
        dnMa[i] = alpha * down[i] + (1 - alpha) * dnMa[i-1]
        rs[i] = abs(upMa[i] - dnMa[i]) / (upMa[i] + dnMa[i]) if upMa[i] + dnMa[i] != 0 else 0 
        rsvaEma[i] = rsvaEma[i-1] + (alpha * (1 + (rs[i] * mult)) * (source[i] - rsvaEma[i-1]))
        slo[i] = source[i] - rsvaEma[i]
        signal[i] = 1 if slo[i] > 0 else -1 
    return rsvaEma, signal
    
    
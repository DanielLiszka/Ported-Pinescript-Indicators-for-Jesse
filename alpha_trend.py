from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/o50NYLAZ-AlphaTrend/#chart-view-comment-form

Alpha Trend
"""

AlphaTrend = namedtuple("AlphaTrend",["AlphaTrend", "PrevAlphaTrend", "signals"])

def alpha_trend(candles: np.ndarray, atr_period:int=14,coeff:float=1, source_type: str = "close", sequential: bool = False) -> AlphaTrend:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    ATR = talib.TRANGE(candles[:,3],candles[:,4],candles[:,2])
    ATR = talib.SMA(ATR,14)
    upT = candles[:,4] - ATR * coeff
    downT = candles[:,3] + ATR * coeff
    MFI = talib.MFI(candles[:,3],candles[:,4],candles[:,2],candles[:,5],timeperiod=atr_period)
    Alpha_Trend,signals = fast_alpha_trend(source,MFI,downT,upT)
    if sequential:
        return AlphaTrend(Alpha_Trend,Alpha_Trend[:-3],signals)
    else:
        return AlphaTrend(Alpha_Trend[-1], Alpha_Trend[-3], signals[-1])

@njit
def fast_alpha_trend(source,MFI,downT,upT):
    Alpha_Trend = np.zeros(source.shape[0])
    signals = np.zeros(source.shape[0])
    for i in range(source.shape[0]):
        if MFI[i] >= 50:
            if (upT[i] < Alpha_Trend[i-1]):
                Alpha_Trend[i] = Alpha_Trend[i-1]
            else:
                Alpha_Trend[i] = upT[i]
        else:
            if (downT[i] > Alpha_Trend[i-1]):
                Alpha_Trend[i] = Alpha_Trend[i-1]
            else:
                Alpha_Trend[i] = downT[i] 
        if Alpha_Trend[i] > Alpha_Trend[i-2] and signals[i-1] != 1:
            signals[i] = 1 
        elif Alpha_Trend[i] < Alpha_Trend[i-2] and signals[i-1] != -1:
            signals[i] = -1 
        else:
            signals[i] = signals[i-1] 
            
    return Alpha_Trend,signals
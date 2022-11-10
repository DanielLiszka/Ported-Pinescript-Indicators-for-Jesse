from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/kRIjThLZ-Trend-Magic/#chart-view-comment-form

Trend Magic
"""

TrendMagic = namedtuple("TrendMagic",["TrendMagic", "signals"])

def trend_magic(candles: np.ndarray, atr_period:int=20,coeff:float=1, source_type: str = "close", sequential: bool = False) -> TrendMagic:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    ATR = talib.TRANGE(candles[:,3],candles[:,4],candles[:,2])
    ATR = talib.SMA(ATR,5)
    upT = candles[:,4] - ATR * coeff
    downT = candles[:,3] + ATR * coeff
    MFI = talib.CCI(candles[:,3],candles[:,4],candles[:,2],timeperiod=atr_period)
    Alpha_Trend,signals = fast_alpha_trend(source,MFI,downT,upT)
    if sequential:
        return TrendMagic(Alpha_Trend,signals)
    else:
        return TrendMagic(Alpha_Trend[-1], signals[-1])

@njit
def fast_alpha_trend(source,MFI,downT,upT):
    Alpha_Trend = np.zeros(source.shape[0])
    signals = np.zeros(source.shape[0])
    for i in range(source.shape[0]):
        if MFI[i] >= 0:
            if (upT[i] < Alpha_Trend[i-1]):
                Alpha_Trend[i] = Alpha_Trend[i-1]
            else:
                Alpha_Trend[i] = upT[i]
        else:
            if (downT[i] > Alpha_Trend[i-1]):
                Alpha_Trend[i] = Alpha_Trend[i-1]
            else:
                Alpha_Trend[i] = downT[i] 
        if MFI[i] >= 0 and signals[i-1] != 1:
            signals[i] = 1 
        elif MFI[i] < 0 and signals[i-1] != -1:
            signals[i] = -1 
        else:
            signals[i] = signals[i-1] 
            
    return Alpha_Trend,signals
from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

Buff = namedtuple('Buff',['Slow_MA','Fast_MA','Signal'])

"""
https://www.tradingview.com/script/htTgm18f-Buff-Averages-CC/#chart-view-comment-form
"""

def buff(candles: np.ndarray, fastLength:int=5, slowLength:int=20,source_type: str = "close", sequential: bool = False) -> Buff:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    slowBuff, fastBuff, slo = func_Buff(source,candles[:,5],candles,fastLength, slowLength)
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
        return Buff(slowBuff,fastBuff,signal)
    else:
        return Buff(slowBuff[-1],fastBuff[-1],signal)

@njit
def func_Buff(source,volume,candles,fastLength,slowLength):
    fastNum = np.zeros(source.shape[0])
    fastDen = np.zeros(source.shape[0])
    fastBuff = np.zeros(source.shape[0])
    slowNum = np.zeros(source.shape[0])
    slowDen = np.zeros(source.shape[0])
    slowBuff = np.zeros(source.shape[0])
    slo = np.zeros(source.shape[0])
    pre_fastNum = np.zeros(source.shape[0])
    for i in range(source.shape[0]):
        pre_fastNum[i] = source[i] * volume[i]
        fastNum[i] = np.sum(pre_fastNum[i-(fastLength-1):i+1])
        fastDen[i] = np.sum(volume[i-(fastLength-1):i+1])
        fastBuff[i] = fastNum[i] / fastDen[i] if fastDen[i] != 0 else 0 
        slowNum[i] = np.sum(pre_fastNum[i-(slowLength-1):i+1])
        slowDen[i] = np.sum(volume[i-(slowLength-1):i+1])
        slowBuff[i] = slowNum[i] / slowDen[i] if slowDen[i] != 0 else 0 
        slo[i] = fastBuff[i] - slowBuff[i] 
    return slowBuff, fastBuff, slo
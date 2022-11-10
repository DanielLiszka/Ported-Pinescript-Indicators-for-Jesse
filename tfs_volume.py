from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config


def tfs(candles: np.ndarray,AvgLen:int=7, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    tfs = func_tfs(source,candles,AvgLen)
    if sequential:
        return tfs
    else:
        return tfs[-1]


@njit(fastmath=True) 
def func_tfs(source,candles,AvgLen):    
    nVolAccum = np.zeros(source.shape[0])
    final = np.zeros(source.shape[0])
    for i in range(AvgLen,source.shape[0]):
        if candles[:,2][i] > candles[:,1][i]:
            nVolAccum[i] = candles[:,5][i]
        elif candles[:,2][i] < candles[:,1][i]:
            nVolAccum[i] = -candles[:,5][i]
        else:
            nVolAccum[i] = nVolAccum[i-1]
        final[i] = np.sum(nVolAccum[i-(AvgLen-1):i+1])/AvgLen
    return final
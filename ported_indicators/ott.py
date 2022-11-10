from collections import namedtuple
import numpy as np
from numba import njit, jit
import talib 
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from typing import Union

OTT = namedtuple('ott', ['var', 'ott'])

def ott(candles: np.ndarray, length: int = 2, period:int = 9,percent: float = 1.4, source_type: str = "close", sequential: bool = False) -> OTT:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    var, ott = var_func(source,length,percent,period)
    if sequential: 
        return OTT(var,ott)
    else:    
        return OTT(var[-1], ott[-1])

#jesse backtest  '2021-01-03' '2021-03-02'
# buy when var > ott and vice versa.
"""
https://www.tradingview.com/script/zVhoDQME/
Only VAR function used 
Same as TV except the two bar delay on OTT plot was removed
"""

@jit(error_model="numpy")
def var_func(source,length,percent,period):
    valpha = 2/(length+1)
    # vUD = np.zeros(source.shape[0])
    vud1 = np.zeros(source.shape[0])
    vudd1 = np.zeros(source.shape[0])
    # vDD = np.zeros(source.shape[0])
    vAR = np.zeros(source.shape[0])
    VAR = np.zeros(source.shape[0])
    fark = np.zeros(source.shape[0])
    longStop = np.zeros(source.shape[0])
    shortStop = np.zeros(source.shape[0]) 
    longStopPrev = np.zeros(source.shape[0])
    shortStopPrev = np.zeros(source.shape[0])
    dir1 = np.zeros(source.shape[0])
    OTT = np.zeros(source.shape[0])
    MT = np.zeros(source.shape[0])
    vCMO = np.zeros(source.shape[0])
    for i in range(period+2,source.shape[0]):
        vud1[i] = source[i] - source[i-1] if source[i] > source[i-1] else 0 
        vudd1[i] = source[i-1] - source[i] if source[i] < source[i-1] else 0 
        vUD = np.sum(vud1[i-(period-1):i+1])
        vDD = np.sum(vudd1[i-(period-1):i+1])
        vCMO[i] = ((vUD-vDD)/(vUD+vDD))
        VAR[i] = (valpha * np.abs(vCMO[i])*source[i]) + (1 - valpha * np.abs(vCMO[i]))*(VAR[i-1])
    for i in range((period+2),source.shape[0]):
        fark[i] = VAR[i] * percent * 0.01 
        longStop[i] = VAR[i] - fark[i]
        shortStop[i] = VAR[i] + fark[i] 
        longStopPrev[i] = longStop[i-1]
        shortStopPrev[i] = shortStop[i-1]
        longStop[i] = np.maximum(longStop[i], longStopPrev[i]) if VAR[i] > longStopPrev[i] else longStop[i] 
        shortStop[i] = np.minimum(shortStop[i], shortStopPrev[i]) if VAR[i] < shortStopPrev[i] else shortStop[i] 
        if VAR[i] > shortStopPrev[i]:
            dir1[i] = 1 
        elif VAR[i] < longStopPrev[i]:
            dir1[i] = -1
        else:
            dir1[i] = dir1[i-1]
        MT[i] = longStop[i] if dir1[i] == 1 else shortStop[i]
        OTT[i] = MT[i] * (200+percent)/200 if VAR[i] > MT[i] else MT[i]*(200-percent)/200
    
    return VAR,OTT


    
    
    
    
    
 
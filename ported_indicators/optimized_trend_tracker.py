from collections import namedtuple
import numpy as np
from numba import njit, jit
import talib 
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from typing import Union

OTT = namedtuple('ott', ['var', 'ott'])

def old_ott(candles: np.ndarray, length:int = 2,period: int = 9, percent: float = 1.4, source_type: str = "close", sequential: bool = False) -> OTT:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    var, ott = VARMA(source,period, percent)
    if sequential: 
        return OTT(var,ott)
    else:    
        return OTT(var[-1], ott[-1])

#jesse backtest  '2021-01-03' '2021-03-02'

"""
https://www.tradingview.com/script/zVhoDQME/
Only VAR function used 
Same as TV except the two bar delay on OTT plot was removed
"""

@jit(error_model='numpy')
def VARMA(source, length, percent):
    vud1 =  np.zeros(source.shape[0])
    vdd1 =  np.zeros(source.shape[0])
    vUD =  np.zeros(source.shape[0])
    vDD =  np.zeros(source.shape[0])
    VAR = np.zeros(source.shape[0])
    vCMO = np.zeros(source.shape[0])
    dir1 = np.zeros(source.shape[0])
    longstop = np.zeros(source.shape[0])
    longstopPrev = np.zeros(source.shape[0])
    shortstop = np.zeros(source.shape[0])
    shortstopPrev = np.zeros(source.shape[0])
    fark = np.zeros(source.shape[0])
    MT = np.zeros(source.shape[0])
    OTT = np.zeros(source.shape[0])
    valpha = 2 / (length+1)
    for i in range(length+1,source.shape[0]):    
        if (source[i] > source[i-1]):
            vud1[i] =  source[i] - source[i-1]
        else: 
            vud1[i] = 0
        if (source[i] < source[i-1]):
            vdd1[i] = source[i-1] - source[i]
        else:
            vdd1[i] = 0 
        vUD[i] = np.sum(vud1[i-(9-1):i+1]) 
        vDD[i] = np.sum(vdd1[i-(9-1):i+1]) 
        vCMO[i] = (vUD[i] - vDD[i])/(vUD[i] + vDD[i])
        VAR[i] = ((valpha*np.abs(vCMO[i])*source[i]) + (1-valpha*np.abs(vCMO[i]))*(VAR[i-1]))
        fark[i] = VAR[i]*percent*0.01
        longstop[i] = VAR[i] - fark[i]
        longstopPrev[i] = longstop[i-1]
        if VAR[i] > longstopPrev[i]:
            longstop[i] = np.maximum(longstop[i], longstopPrev[i])
        else:
            longstop[i] = longstop[i]
        shortstop[i] = VAR[i] + fark[i] 
        shortstopPrev[i] = shortstop[i-1] 
        if VAR[i] < shortstopPrev[i]:
            shortstop[i] = np.minimum(shortstop[i], shortstopPrev[i])
        else:
            shortstop[i] = shortstop[i] 
        if dir1[i-1] == -1 and VAR[i] > shortstopPrev[i]:
            dir1[i] = 1 
        elif dir1[i-1] == 1 and VAR[i] < longstopPrev[i]:
            dir1[i] = -1 
        else:
            dir1[i] = dir1[i-1] 
        if dir1[i] == 1:
            MT[i] = longstop[i] 
        else:
            MT[i] = shortstop[i] 
        if VAR[i] > MT[i]:
            OTT[i] = MT[i]*(200 + percent)/200 
        else:
            OTT[i] = MT[i]*(200 - percent)/200    
    return VAR, OTT
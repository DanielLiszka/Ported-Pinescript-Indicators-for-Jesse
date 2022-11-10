from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/1bmPQFS6-Volume-Spike-Retracement/#chart-view-comment-form

Volume Spike Retracement
"""
VSR = namedtuple("VSR",["bearvolprice","bullvolprice","buysignal","sellsignal"])

def vsr(candles: np.ndarray, length:int=89,mult:float=0.5, source_type: str = "close", sequential: bool = False) -> VSR:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    hvs = talib.MAX(candles[:,5],length)
    abs1 = candles[:,5] * 100 / hvs * 4 / 5
    smoothing = talib.EMA(abs1, 21)
    equal = abs1 - smoothing
    limit = talib.MAX(equal,length) * mult
    bearvolprice,bullvolprice = vsr_func(source,candles,length,mult,limit,equal)
    sell = True if bearvolprice[-1] != bearvolprice[-2] else False 
    buy = True if bullvolprice[-1] != bullvolprice[-2] else False 
    if sequential:
        return VSR(bearvolprice,bullvolprice,buysignal,sellsignal)
    else:
        return VSR(bearvolprice[-1], bullvolprice[-1],buy,sell)

@njit
def vsr_func(source,candles,length,mult,limit,equal):
    cum = np.zeros(source.shape[0])
    beardir = np.zeros(source.shape[0])
    bulldir = np.zeros(source.shape[0])
    bearvol = np.zeros(source.shape[0])
    bullvol = np.zeros(source.shape[0])
    bearvolprice = np.zeros(source.shape[0])
    bullvolprice = np.zeros(source.shape[0])
    for i in range(source.shape[0]):
        cum[i] = 1 if equal[i] > 0 and equal[i] >= limit[i] else 0 
        beardir[i] = 1 if candles[:,2][i] < candles[:,1][i] else 0 
        bulldir[i] = 1 if candles[:,2][i] > candles[:,1][i] else 0
        bearvol[i] = -1 if beardir[i] == 1 and cum[i] == 1 else 0 
        bullvol[i] = 1 if bulldir[i] == 1 and cum[i] == 1 else 0 
        bearvolprice[i] = candles[:,3][i] if bearvol[i] == -1 else bearvolprice[i-1]
        bullvolprice[i] = candles[:,4][i] if bullvol[i] == 1 else bullvolprice[i-1]
    return bearvolprice, bullvolprice
            
        

import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple

#jesse backtest  '2021-01-03' '2021-03-02'

OTTO = namedtuple('OTTO',['price_source','OTTO'])

'''
https://www.tradingview.com/script/W3BqP7Nq-Optimized-Trend-Tracker-Oscillator-OTTO/#chart-view-comment-form
optimized trend tracker oscillator. Two Bar delay removed
''' 
  
def otto(candles: np.ndarray, length:int=2, varlength:int=9,percent:float=0.6,flength:int=10,slength:int=25,coco:int=100000,source_type: str = "close", sequential: bool = False ) -> OTTO:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    hott,lott = otto_func(source,candles,length,slength,flength,coco,percent,varlength)
    if sequential:
        return OTTO(lott, hott)
    else:
        return OTTO(lott[-1], hott[-1])

@jit(error_model="numpy")
def otto_func(source,candles,length, slength,flength,coco,percent,varlength):
    valpha = 2 / (length+1)
    valpha2 = 2 / (slength/2 + 1)
    valpha3 = 2 / (slength + 1)
    valpha4 = 2 / (slength * flength + 1)
    src = np.zeros(source.shape[0])
    vud1 = np.zeros(source.shape[0])
    vdd1 = np.zeros(source.shape[0])
    vUD = np.zeros(source.shape[0])
    vDD = np.zeros(source.shape[0])
    vCMO = np.zeros(source.shape[0])
    VAR2 = np.zeros(source.shape[0])
    VAR3 = np.zeros(source.shape[0])
    VAR4 = np.zeros(source.shape[0])
    new_vud1 = np.zeros(source.shape[0])
    new_vdd1 = np.zeros(source.shape[0])
    new_vDD = np.zeros(source.shape[0])
    new_vUD = np.zeros(source.shape[0])
    new_vCMO = np.zeros(source.shape[0])
    new_VAR = np.zeros(source.shape[0])
    dir1 = np.zeros(source.shape[0])
    longstop = np.zeros(source.shape[0])
    longstopPrev = np.zeros(source.shape[0])
    shortstop = np.zeros(source.shape[0])
    shortstopPrev = np.zeros(source.shape[0])
    fark = np.zeros(source.shape[0])
    MT = np.zeros(source.shape[0])
    OTT = np.zeros(source.shape[0])
    for i in range(slength+1,source.shape[0]):    
        if (source[i] > source[i-1]):
            vud1[i] =  source[i] - source[i-1]
        else: 
            vud1[i] = 0
        if (source[i] < source[i-1]):
            vdd1[i] = source[i-1] - source[i]
        else:
            vdd1[i] = 0 
        vUD[i] = np.sum(vud1[i-(varlength-1):i+1]) 
        vDD[i] = np.sum(vdd1[i-(varlength-1):i+1]) 
        vCMO[i] = (vUD[i] - vDD[i])/(vUD[i] + vDD[i])
        VAR2[i] = ((valpha2*np.abs(vCMO[i])*source[i]) + (1-valpha2*np.abs(vCMO[i]))*(VAR2[i-1]))
        VAR3[i] = ((valpha3*np.abs(vCMO[i])*source[i]) + (1-valpha3*np.abs(vCMO[i]))*(VAR3[i-1]))
        VAR4[i] = ((valpha4*np.abs(vCMO[i])*source[i]) + (1-valpha4*np.abs(vCMO[i]))*(VAR4[i-1]))
        src[i] = VAR2[i] / (VAR3[i] - VAR4[i] + coco)
        new_vud1[i] = src[i] - src[i-1] if  (src[i] > src[i-1]) else 0
        new_vdd1[i] = src[i-1] - src[i] if (src[i] < src[i-1]) else 0 
        new_vUD[i] = np.sum(vud1[i-(varlength-1):i+1]) 
        new_vDD[i] = np.sum(vdd1[i-(varlength-1):i+1]) 
        new_vCMO[i] = (new_vUD[i] - new_vDD[i])/(new_vUD[i] + new_vDD[i])
        new_VAR[i] = ((valpha*np.abs(new_vCMO[i])*src[i]) + (1-valpha*np.abs(new_vCMO[i]))*(new_VAR[i-1]))
        fark[i] = new_VAR[i]*percent*0.01
        longstop[i] = new_VAR[i] - fark[i]
        longstopPrev[i] = longstop[i-1]
        if new_VAR[i] > longstopPrev[i]:
            longstop[i] = np.maximum(longstop[i], longstopPrev[i])
        else:
            longstop[i] = longstop[i]
        shortstop[i] = new_VAR[i] + fark[i] 
        shortstopPrev[i] = shortstop[i-1] 
        if new_VAR[i] < shortstopPrev[i]:
            shortstop[i] = np.minimum(shortstop[i], shortstopPrev[i])
        else:
            shortstop[i] = shortstop[i] 
        if dir1[i-1] == -1 and new_VAR[i] > shortstopPrev[i]:
            dir1[i] = 1 
        elif dir1[i-1] == 1 and new_VAR[i] < longstopPrev[i]:
            dir1[i] = -1 
        else:
            dir1[i] = dir1[i-1] 
        if dir1[i] == 1:
            MT[i] = longstop[i] 
        else:
            MT[i] = shortstop[i] 
        if new_VAR[i] > MT[i]:
            OTT[i] = MT[i]*(200 + percent)/200 
        else:
            OTT[i] = MT[i]*(200 - percent)/200   
    return OTT, new_VAR
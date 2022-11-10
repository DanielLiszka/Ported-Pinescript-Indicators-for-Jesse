from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple 

ATR_LADDER = namedtuple('ATR_LADDER',['Direction', 'Supertrend_Stop',])

def atr_ladder(candles: np.ndarray, multiplier:float=4,malength:int=20,matype:str="ema", delayed:bool=False,waitForClose:bool=False, source_type: str = "close", sequential: bool = False) -> ATR_LADDER:
    candles = slice_candles(candles, sequential)
    source = candles[:,2] #get_candle_source(candles, source_type=source_type)
    tr = talib.TRANGE(candles[:,3],candles[:,4],candles[:,2])
    dir1,stop = func_atr_ladder(source,candles,multiplier,malength,matype,tr,delayed,waitForClose)
    
    if sequential:
        return ATR_LADDER(dir1,stop)
    else:
        return ATR_LADDER(dir1[-1],stop[-1])


#not accurate, dir1 is unchanging and positiveAtr,negativeAtr are not accurate
@njit(fastmath=True) 
def func_atr_ladder(source, candles, multiplier, malength, matype,tr,delayed,waitForClose):
    dir1 = np.zeros(source.shape[0])
    targetReached = 0
    positiveTr = np.zeros(source.shape[0])
    negativeTr  = np.zeros(source.shape[0])
    positiveAtr  = np.zeros(source.shape[0])
    negativeAtr  = np.zeros(source.shape[0])
    buyStopDiff  = np.zeros(source.shape[0])
    buyStopCurrent  = np.zeros(source.shape[0])
    sellStopDiff  = np.zeros(source.shape[0])
    sellStopCurrent  = np.zeros(source.shape[0])
    highConfirmation  = np.zeros(source.shape[0])
    lowConfirmation  = np.zeros(source.shape[0])
    buystop = np.zeros(source.shape[0]) 
    sellstop  = np.zeros(source.shape[0])
    stop = np.zeros(source.shape[0])
    for i in range(malength+1,source.shape[0]):
        if i == (malength+1) :
            dir1[i] = 1
            dir1[i-1] = 1 
        dir1[i] = dir1[i-1]
        if candles[:,1][i] < candles[:,2][i]:
            positiveTr[i] = tr[i]
            negativeTr[i] = negativeTr[i-1]
        else:
            positiveTr[i] = positiveTr[i-1]
            negativeTr[i] = tr[i] 
        positiveAtr[i] = pine_ema(source,positiveTr,malength)[i]
        negativeAtr[i] = pine_ema(source,negativeTr,malength)[i]
        buyStopDiff[i] = negativeAtr[i] * multiplier
        sellStopDiff[i] = positiveAtr[i] * multiplier
        buyStopDiff[i] = np.minimum(buyStopDiff[i],buyStopDiff[i-1]) if dir1[i] == 1 else buyStopDiff[i]
        sellStopDiff[i] = np.minimum(sellStopDiff[i], sellStopDiff[i-1]) if dir1[i] == -1 else sellStopDiff[i]
        buystop[i] = candles[:,4][i] - buyStopDiff[i] 
        sellstop[i] = candles[:,3][i] + sellStopDiff[i]
        buyStopCurrent[i] = candles[:,4][i] - buyStopDiff[i]
        sellStopCurrent[i] = candles[:,3][i] + sellStopDiff[i] 
        highConfirmation[i] = source[i] if waitForClose else candles[:,3][i]
        lowConfirmation[i] = source[i] if waitForClose else candles[:,4][i] 
        if (dir1[i] == 1 and lowConfirmation[i-1] < buystop[i-1]):
            dir1[i] = -1
        else:
            if (dir1[i] == -1 and highConfirmation[i-1] > sellstop[i-1]):
                dir1[i] = 1 
            else:
                dir1[i] = dir1[i-1] 
        targetReached = 1 if (dir1[i] == 1  and highConfirmation[i-1] >= sellstop[i-1]) or (dir1[i] == -1 and lowConfirmation[i-1] <= buystop[i-1]) or (not delayed) else 0
        if (dir1[i] == 1):
            if targetReached == 1:
                buystop[i] = np.maximum(buystop[i],buyStopCurrent[i])
                sellstop[i] = sellStopCurrent[i]
            else:
                buystop[i] = buystop[i]
                sellstop[i] = np.minimum(sellstop[i],(candles[:,3][i] + sellStopDiff[i]/2))
        elif (dir1[i] == -1):  
            if targetReached == 1:
                buystop[i] = buyStopCurrent[i]
                sellstop[i] = np.minimum(sellstop[i],sellStopCurrent[i])
            else:
                buystop[i] = np.maximum(buystop[i], (candles[:,4][i] - buyStopDiff[i]/2))
                sellstop[i] = sellstop[i]
        stop[i] = buystop[i] if dir1[i] == 1 else sellstop[i]      
    return dir1,stop
                
    
    
@njit(fastmath=True)
def pine_ema(source1, source2, length):
    alpha = 2 / (length + 1)
    sum1 = np.zeros(source1.shape[0])
    for i in range(10,source1.shape[0]):
        #sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    return sum1  
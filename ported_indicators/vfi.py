from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

VFI = namedtuple('VFI',['vfi','vfima'])

"""
https://www.tradingview.com/script/MhlDpfdS-Volume-Flow-Indicator-LazyBear/

difference in candle volume makes it inaccurate like most volume based indicators
"""

def vfi(candles: np.ndarray, length:int=130,coef:float=0.2,vcoef:float=2.5,signalLength:int=5,smoothVFI:bool=False,source_type: str = "hlc3", sequential: bool = False) -> VFI:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    sma2 = talib.SMA(candles[:,5],length)
    vcp = vfi_func(source,candles,length,coef,vcoef,signalLength,sma2)
    vfi = pine_sma(source,vcp,3) if smoothVFI else vcp
    vfima = pine_ema(source,vfi,signalLength)
    d = vfi-vfima
    
    if sequential:
        return VFI(vfi,vfima)
    else:
        return VFI(vfi[-1],vfima[-1])


@njit(fastmath=True) 
def vfi_func(source,candles,length,coef,vcoef,signal,sma2):
    inter = np.zeros(source.shape[0])
    vinter = np.zeros(source.shape[0])
    cutoff = np.zeros(source.shape[0])
    sma1 = np.zeros(source.shape[0])
    vave = np.zeros(source.shape[0])
    vcp = np.zeros(source.shape[0])
    vfi = np.zeros(source.shape[0])
    vmax = np.zeros(source.shape[0])
    vc = np.zeros(source.shape[0])
    mf = np.zeros(source.shape[0])
    for i in range(length,source.shape[0]):
        inter[i] = np.log(source[i]) - np.log(source[i-1])
        # vinter = std2(inter,sma1,30)
        sum2 = 0.0
        for j in range(30):
            sum2 = sum2 + inter[i-j]/30
        sma1[i] = sum2 
        sum1= 0.0
        sum2= 0.0
        for j in range(30):
            sum1 = (inter[i-j] + -(sma1[i]))
            sum2 = sum2 + sum1 * sum1 
        vinter[i] = np.sqrt(sum2/30)
        cutoff[i] = coef * vinter[i] * candles[:,2][i]
        vave[i] = sma2[i-1]
        vmax[i] = vave[i] * vcoef
        if candles[:,5][i] < vmax[i]:
            vc[i] = candles[:,5][i]
        else:
            vc[i] = vmax[i] 
        mf[i] = source[i] - source[i-1] 
        if mf[i] > cutoff[i]:
            vcp[i] = vc[i] 
        elif (mf[i] < -cutoff[i]):
            vcp[i] = -vc[i] 
        else:
            vcp[i] = 0 
        vfi[i] = (np.sum(vcp[i-(length+1):i+1]))/vave[i]
    
    return vfi

@njit(fastmath=True)
def std2(source,avg,per):
    std1 = np.full_like(source,0)
    for i in range(source.shape[0]):
        sum1 = 0.0
        sum2 = 0.0
        for j in range(per):
            sum1 = (source[i-j] + -avg[i])
            sum2 = sum2 + sum1 * sum1 
        std1[i] = np.sqrt(sum2 / per)
    return std1
    
@njit(fastmath=True)
def pine_ema(source1, source2, length):
    alpha = 2 / (length + 1)
    sum1 = np.full_like(source1,0)
    for i in range(0,source1.shape[0]):
        sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    return sum1 

@njit(fastmath=True)
def pine_sma(source1,source2,length):
    sum1 = np.full_like(source1,0)
    for i in range(source1.shape[0]):
        sum2 = 0.0
        for j in range(length):
            sum2 = sum2 + source2[i-j]/length
        sum1[i] = sum2 
    return sum1 
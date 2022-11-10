from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple 

SAMA = namedtuple('SAMA', ['signal','top_tp', 'bottom_tp'])

"""
https://www.tradingview.com/script/Ies7Tygo-Slope-Adaptive-Moving-Average-MZ-SAMA/
"""

def sama(candles: np.ndarray,mult:float=2.5,length:int=200,majLength:int=14,minLength:int=6,slopeLength:int=34,slopeInRange:int=25,flat:int=17,  source_type: str = "close", sequential: bool = False) -> SAMA:
    candles = candles[-800:] #slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    ma,longsignal, shortsignal = func_SAMA(source,candles,length,majLength,minLength,slopeLength,slopeInRange,flat)
    signal = 0
    atr = talib.ATR(candles[:,3],candles[:,4],candles[:,2],timeperiod=40)
    utl = ma[-1] + mult * atr[-1]
    ltl = ma[-1] - mult * atr[-1]
    ttl = ma[-1] + (mult*2) * atr[-1]
    btl = ma[-1] - (mult*2) * atr[-1]
    stp_utl = ma[-1] + (mult/2) * atr[-1]
    stp_ltl = ma[-1] - (mult/2) * atr[-1]
    top_out = True if candles[:,3][-1] > ttl and not candles[:,3][-2] > ttl else False
    bot_out = True if candles[:,4][-1] < btl and not candles[:,4][-2] < btl else False
    if longsignal == 1 and shortsignal == 0:
        signal = 1 
    if longsignal == 0 and shortsignal == 1:
        signal = -1 
    if sequential:
        return signal,top_out,bot_out
    else:
        return signal,top_out,bot_out



@njit(fastmath=True) 
def func_SAMA(source,candles,length,majLength,minLength,slopeLength,slopeInRange,flat):
    hh = np.zeros(source.shape[0])
    ll = np.zeros(source.shape[0])
    mult = np.zeros(source.shape[0])
    final = np.zeros(source.shape[0])
    ma = np.zeros(source.shape[0])
    highestHigh = np.zeros(source.shape[0])
    lowestLow = np.zeros(source.shape[0])
    slope_range = np.zeros(source.shape[0])
    dt = np.zeros(source.shape[0])
    c = np.zeros(source.shape[0])
    xAngle = np.zeros(source.shape[0])
    maAngle = np.zeros(source.shape[0])
    _up = np.zeros(source.shape[0])
    _down = np.zeros(source.shape[0])
    swing = np.zeros(source.shape[0])
    longsignal = 0
    shortsignal = 0
    minAlpha = 2 / (minLength + 1)
    majAlpha = 2 / (majLength + 1)
    for i in range(length+1,source.shape[0]):    
        hh[i] = np.amax(candles[i-(length):i+1:,3])
        ll[i] = np.amin(candles[i-(length):i+1:,4])
        mult[i] = np.abs(2 * source[i] - ll[i] - hh[i])/ (hh[i] - ll[i]) if hh[i] - ll[i] != 0 else 0 
        final[i] = mult[i] * (minAlpha - majAlpha) + majAlpha
        ma[i] = ma[i-1] + np.power(final[i],2) * (source[i] - ma[i-1])
    for i in range((source.shape[0]-(slopeLength+1)),source.shape[0]):
        swing[i] = swing[i-1]
        highestHigh[i] = np.amax(candles[i-(slopeLength-1):i+1,3])
        lowestLow[i] = np.amin(candles[i-(slopeLength-1):i+1,4])
        slope_range[i] = slopeInRange / (highestHigh[i] - lowestLow[i]) * lowestLow[i]
        dt[i] = (ma[i-2] - ma[i]) / source[i] * slope_range[i] 
        c[i] = np.sqrt(1 + dt[i] * dt[i])
        xAngle[i] = np.round(180 * np.arccos(1/c[i]) / np.pi)
        maAngle[i] = -xAngle[i] if dt[i] > 0 else xAngle[i] 
        _up[i] = 1 if maAngle[i] > flat else 0
        _down[i] = -1 if maAngle[i] <= -flat else 0 
        if (_up[i] == 1 and _up[i-1] != 1) and swing[i] <= 0:
            swing[i] = 1 
        if (_down[i] == -1 and _down[i-1] != -1) and swing[i] >= 0:
            swing[i] = -1 
        longsignal = 1 if swing[i] == 1 and swing[i-1] != 1 else 0
        shortsignal = 1 if swing[i] == -1 and swing[i-1] != -1 else 0
    return ma,longsignal, shortsignal

        
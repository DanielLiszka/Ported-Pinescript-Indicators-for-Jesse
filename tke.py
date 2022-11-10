from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/Pcbvo0zG/

mean of seven indicators : rsi , stoch, ultimate oscillator, mfi, williams %R, Momentum, CCI
"""
TKE = namedtuple("TKE",["tke_line","ema_line"])

def tke(candles: np.ndarray, period:int=14, emaperiod: int =5, length7:int=7,length14:int=14,length28:int=28,source_type: str = "close", sequential: bool = False) -> TKE:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    momentum = pine_mom(source,period)
    cci = talib.CCI(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod=period)
    rsi = talib.RSI(source,timeperiod=period)
    willr = talib.WILLR(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod=period)
    mfi = talib.MFI(candles[:, 3], candles[:, 4], candles[:, 2], candles[:, 5], timeperiod=period)
    candles_close = candles[:, 2]
    candles_high = candles[:, 3]
    candles_low = candles[:, 4]
    hh = talib.MAX(candles_high, period)
    ll = talib.MIN(candles_low, period)
    stoch = 100 * (candles_close - ll) / (hh - ll)
    ult = talib.ULTOSC(candles[:, 3], candles[:, 4], candles[:, 2], timeperiod1=length7, timeperiod2=length14,
                       timeperiod3=length28)
    tkeline = (ult + mfi + momentum + cci + rsi + willr + stoch)/7
    emaline = pine_ema(source,tkeline,emaperiod)

    if sequential:
        return TKE(tkeline,emaline)
    else:
        return TKE(tkeline[-1],emaline[-1])

@njit
def pine_mom(source1,length):
    out = np.zeros(source1.shape[0])
    for i in range(source1.shape[0]):
        out[i] = source1[i] / source1[i-length] * 100
    return out 
    
@njit
def pine_ema(source1, source2, length):
    alpha = 2 / (length + 1)
    sum1 = np.full_like(source1,0)
    for i in range(0,source1.shape[0]):
        sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    return sum1 

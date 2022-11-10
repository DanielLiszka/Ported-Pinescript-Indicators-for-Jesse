from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d
from collections import namedtuple

#jesse backtest '2021-01-03' '2021-03-02'

"""
https://www.tradingview.com/script/o4zx8PNC-RSX-D-ID-AC-P/
more testing needed
Divergences and Pivots included
"""

def jmarsx(candles: np.ndarray, xbars:int=90, length: int= 14, periodA: int= 20, periodB : int=40, periodC: int=80, periodD:int=10, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    rsx = fast_rsx(source,candles,length)
    lsmaA = talib.LINEARREG(rsx, timeperiod=periodA)
    lsmaB = talib.LINEARREG(rsx, timeperiod=periodB)
    lsmaC = talib.LINEARREG(rsx, timeperiod=periodC)
    lsmaD = talib.LINEARREG(((lsmaA + lsmaB + lsmaC)/3), timeperiod=periodD)
    rsx_div_l, rsx_div_h, lsmaD_div_l, lsmaD_div_h, divbear,divbull,pivoth,pivotl = otherfunction(source,candles,rsx,xbars,lsmaD)
    if sequential:
        return rsx[-2:]
    else:
        return rsx[-2:]  
@njit        
def fast_rsx(source,candles,length):  
    f8 = np.full_like(source,0)
    f10 = np.full_like(source,0)
    v8 = 0 
    f18 = 0 
    f20 = 0 
    f28 = 0
    f30 = 0 
    vC = 0 
    f38 = 0 
    f40 = 0
    f48 = 0
    f50 = 0
    v14 = 0
    f58 = 0
    f60 = 0
    v18 = 0
    f68 = 0
    f70 = 0
    v1C = 0
    f78 = 0
    f80 = 0
    v20 = 0
    v4 = 0
    rsx = np.full_like(source, 0)
    for i in range(source.shape[0]):
        f8[i] = 100*source[i] 
        f10[i] = f8[i-1]
        v8 = f8[i] - f10[i] 
        f18 = 3 / (length + 2)
        f20 = 1 - f18
        f28 = f20 * f28 + f18 * v8
        f30 = f18 * f28 + f20 * f30
        vC = f28 * 1.5 - f30 * 0.5 
        f38 = f20 * f38 + f18 * vC
        f40 = f18 * f38 + f20 * f40 
        v10 = f38 * 1.5 - f40 * 0.5
        f48 = f20 * f48 + f18 * v10 
        f50 = f18 * f48 + f20 * f50 
        v14 = f48 * 1.5 - f50 * 0.5 
        f58 = f20 * f58 + f18 * np.abs(v8) 
        f60 = f18 * f58 + f20 * f60 
        v18 = f58 * 1.5 - f60 * 0.5
        f68 = f20 * f68 + f18 * v18 
        f70 = f18 * f68 + f20 * f70 
        v1C = f68 * 1.5 - f70 * 0.5
        f78 = f20 * f78 + f18 * v1C
        f80 = f18 * f78 + f20 * f80 
        v20 = f78 * 1.5 - f80 * 0.5 
        v4 = (v14 / v20 + 1)* 50 if v20 > 0 else 50 
        if v4 > 100:
            rsx[i] = 100 
        elif v4 < 0: 
            rsx[i] = 0 
        else:
            rsx[i] = v4  
    return rsx 

@njit        
def otherfunction(source,candles,rsx,xbars,lsmaD):
    highindex1 = np.full_like(source,0)
    lowindex1 = np.full_like(source,0)
    max1 = np.full_like(source,0)
    min1 = np.full_like(source,0)
    max_rsi = np.full_like(source,0)
    min_rsi = np.full_like(source,0)
    pivoth = np.full_like(source,0)
    pivotl = np.full_like(source,0)
    divbear = np.full_like(source,0)
    divbull = np.full_like(source,0)
    f_top_fractalk = np.full_like(source,0)
    f_top_fractalk2 = np.full_like(source,0)
    f_bottom_fractalk = np.full_like(source,0)
    f_bottom_fractalk2 = np.full_like(source,0)
    f_fractalizek = np.full_like(source,0)
    f_fractalizek2 = np.full_like(source,0)
    fractal_top = np.full_like(source,0)
    fractal_top2 = np.full_like(source,0)
    fractal_bot = np.full_like(source,0)
    fractal_bot2 = np.full_like(source,0)
    for i in range(source.shape[0]):
        highestvalue = 0
        highindex = 0
        lowestvalue = 0 
        lowindex = 0 
        for j in range(0,xbars):
            if highestvalue <= (rsx[i-j]):
                highestvalue = rsx[i-j]
                highindex = -j 
            if lowestvalue >= rsx[i-j]:
                lowestvalue = rsx[i-j]
                lowindex = -j 
        highindex1[i] = np.abs(highindex)
        lowindex1[i] = np.abs(lowindex)
        if highindex1[i] == 0:
            max_rsi[i] = rsx[i] 
            max1[i] = candles[:,2][i] 
        elif np.isnan(max1[i-1]):
            max1[i] = candles[:,2][i] 
        else:
            max1[i] = max1[i-1] 
        if highindex1[i] == 0: 
            max_rsi[i] = rsx[i] 
        elif np.isnan(max_rsi[i-1]):
            max_rsi[i] = rsx[i] 
        else:
            max_rsi[i] = max_rsi[i-1] 
        if lowindex1[i] == 0:   
            min1[i] = candles[:,2][i]
        elif np.isnan(min1[i-1]):
            min1[i] = candles[:,2][i] 
        else:
            min1[i] = min1[i-1] 
        if lowindex1[i] == 0:
            min_rsi[i] = rsx[i] 
        elif np.isnan(min_rsi[i-1]):
            min_rsi[i] = rsx[i] 
        else:
            min_rsi[i] = min_rsi[i-1] 
        if candles[:,2][i] > max1[i]:
            max1[i] = candles[:,2][i] 
        if rsx[i] > max_rsi[i]:
            max_rsi[i] = rsx[i] 
        if candles[:,2][i] < min1[i]:
            min1[i] = candles[:,2][i] 
        if rsx[i] < min_rsi[i]:
            min_rsi[i] = rsx[i] 
        if (max_rsi[i] == max_rsi[i-2] and max_rsi[i-2] != max_rsi[i-3]):
            pivoth[i] = 1 
        else:
            pivoth[i] = 0
        if (min_rsi[i] == min_rsi[i-2] and min_rsi[i-2] != min_rsi[i-3]):
            pivotl[i] = 1 
        else:
            pivotl[i] = 0 
        if max1[i-1] > max1[i-2] and rsx[i-1] < max_rsi[i] and rsx[i] <= rsx[i-1]:
            divbear[i] = 1 
        if min1[i-1] < min1[i-2] and rsx[i-1] > min_rsi[i] and rsx[i] >= rsx[i-1]:
            divbull[i] = 1 
        f_top_fractalk[i] = 1 if rsx[i-4] < rsx[i-2] and rsx[i-3] < rsx[i-2] and rsx[i-2] > rsx[i-1] and rsx[i-2] > rsx[i] else 0  
        f_bottom_fractalk[i] = 1 if rsx[i-4] > rsx[i-2] and rsx[i-3] > rsx[i-2] and rsx[i-2] < rsx[i-1] and rsx[i-2] < rsx[i] else 0 
        if f_top_fractalk[i] == 1:
            f_fractalizek[i] = 1 
        elif f_bottom_fractalk[i] == 1:
            f_fractalizek[i] = -1
        else:
            f_fractalizek[i] = 0 
        f_top_fractalk2[i] = 1 if lsmaD[i-4] < lsmaD[i-2] and lsmaD[i-3] < lsmaD[i-2] and lsmaD[i-2] > lsmaD[i-1] and lsmaD[i-2] > lsmaD[i] else 0 
        f_bottom_fractalk2[i] = 1 if lsmaD[i-4] > lsmaD[i-2] and lsmaD[i-3] > lsmaD[i-2] and lsmaD[i-2] < lsmaD[i-1] and lsmaD[i-2] < lsmaD[i] else 0 
        if f_top_fractalk2[i] == 1:
            f_fractalizek2[i] = 1 
        elif f_bottom_fractalk2[i] == 1:
            f_fractalizek2[i] = -1
        else:
            f_fractalizek2[i] = 0 
        fractal_top[i] = rsx[i-2] if f_fractalizek[i] > 0 else np.nan
        fractal_bot[i] = rsx[i-2] if f_fractalizek[i] < 0 else np.nan 
        fractal_top2[i] = lsmaD[i-2] if f_fractalizek2[i] > 0 else np.nan
        fractal_bot2[i] = lsmaD[i-2] if f_fractalizek2[i] < 0 else np.nan 
    return fractal_bot, fractal_top, fractal_bot2, fractal_top2, divbear, divbull, pivoth, pivotl
        
        
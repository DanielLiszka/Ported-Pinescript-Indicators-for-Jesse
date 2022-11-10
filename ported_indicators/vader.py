from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from collections import namedtuple
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/7s0lx2Bw-RedK-Volume-Accelerated-Directional-Energy-Ratio-RedK-VADER/

RedK VADER 

Full volume is accurate
"""

VADER = namedtuple("VADER",["supply_energy","demand_energy","signal","macd"])

def vader(candles: np.ndarray, length:int=10,DER_avg:int=5,MA_Type:str="WMA",smooth:int=3,senti:int=20,v_calc:str="full",vlookbk:int=20, source_type: str = "close", sequential: bool = False) -> VADER:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    if v_calc == "relative":
        hh = talib.MAX(candles[:,5], vlookbk)
        ll = talib.MIN(candles[:,5], vlookbk)
        v_stoch = ( 100 * (candles[:,5] - ll) / (hh - ll))/100
    else:
        v_stoch = candles[:,5]
    R = ((talib.MAX(candles[:,3],2)-talib.MIN(candles[:,4],2))/2)
    sr = np.diff(source,n=1) 
    sr = np.insert(sr,0,0)
    sr = sr / R
    rsr = np.clip(sr,-1,1)
    c = rsr * v_stoch
    c = np.nan_to_num(c,0)
    c_plus = np.clip(c,0,999999999)
    c_minus = np.clip(-c,0,99999999)
    if MA_Type ==  "WMA":
       avg_vola = talib.WMA(v_stoch,length)
       dem = talib.WMA(c_plus,length) / avg_vola
       sup = talib.WMA(c_minus,length) / avg_vola
    else:
        avg_vola = talib.EMA(v_stoch,length)
        dem = talib.EMA(c_plus,length) / avg_vola
        sup = talib.EMA(c_minus,length) / avg_vola
    adp = 100 * talib.WMA(dem, DER_avg)
    asp = 100 * talib.WMA(sup,DER_avg)
    anp = adp - asp
    anp_s = talib.WMA(anp,smooth)

    
    if sequential:
        return VADER(asp,adp,anp_s,anp)
    else:
        return VADER(asp[-1],adp[-1],anp_s[-1],anp[-1])

@njit 
def pine_wma(source1,source2,length):
    res = np.zeros(source1.shape[0])
    for i in range(source1.shape[0]):
        weight = 0.0
        norm = 0.0 
        sum1 = 0.0
        for j in range(length):
            weight = (length - j)*length
            norm = norm + weight 
            sum1 = sum1 + source2[i-j] * weight
        res[i] = sum1/norm 
    return res 
    
@njit 
def pine_ema(source1, source2, length):
    alpha = 2 / (length + 1)
    sum1 = np.zeros(source1.shape[0])
    for i in range(10,source1.shape[0]):
        sum1[i-1] = 0 if np.isnan(sum1[i-1]) else sum1[i-1]
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    return sum1 
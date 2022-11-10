from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config, same_length, get_candle_source, slice_candles, np_shift
from collections import namedtuple

#jesse backtest  '2021-01-03' '2021-03-02'

'''
https://www.tradingview.com/script/AZn3t39d-Godmode-Oscillator-3-2/#chart-view-comments
MoneyFlow Removed and csi changed to use source as input
''' 
GMODE = namedtuple('GMODE',['godmode','signal','wave'])

def godmode(candles: np.ndarray, n1: int= 17, n2:int=6,n3:int=4,sig:int=6,topthreshold:int=85, bottomthreshold:int=15, source_type: str = "close", sequential: bool = False ) -> GMODE:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    tci = talib.EMA((source - talib.EMA(source,n1))/(0.025*talib.EMA(np.abs(source - talib.EMA(source,n1)),n1)),n2)+50
    willy = 60 * (source - talib.MAX(source,n2))/(talib.MAX(source,n2) - talib.MIN(source,n2))+80
    pc = source - np_shift(source,1,np.nan)
    double_smoothed_pc = talib.EMA(talib.EMA(pc,n1),n2)
    double_smoothed_pc_abs = talib.EMA(talib.EMA(np.abs(pc),n1),n2)
    tsi_value = double_smoothed_pc / double_smoothed_pc_abs
    csi = (talib.RSI(source,n3) + tsi_value*50+50)/2
    godmode = (tci + csi + willy) / 3 
    signal = talib.SMA(godmode,sig)
    wave = talib.EMA(((godmode-signal)*2+50),n3)
   # res = fast_godmode(source,candles,n1,n2,n3)
    if sequential: 
        return GMODE(godmode,signal,wave)
    else:
        return GMODE(godmode[-1],signal[-1],wave[-1])
	
	
# @jit(error_model='numpy')
# def fast_godmode(source,candles,n1,n2,n3):
    # tci = np.full_like(source,0)
    # rsi = np.full_like(source,0)
    # emasource = np.full_like(source,0)
    # tci = np.full_like(source,0)
    # alpha1 = 2 / (n1 + 1) 
    # alpha2 = 2 / (n2 + 1)
    # ema0 = np.full_like(source,0)
    # ema1 = np.full_like(source,0)
    # ema2 = np.full_like(source,0)
    # maxema = np.full_like(source,0)
    # willy = np.full_like(source,0)
    # rsialpha = 1 / n3 
    # pc = np.full_like(source,0)
    # double_smoothed_pc = np.full_like(source,0)
    # double_smoothed_pc_abs = np.full_like(source,0)
    # tsi_value = np.full_like(source,0)
    # csi = np.full_like(source,0)
    # u = np.full_like(source, 0)
    # d = np.full_like(source, 0)
    # rs = np.full_like(source, 0)
    # res = np.full_like(source, 0)
    # sumation1 = np.full_like(source, 1)
    # sumation2 = np.full_like(source, 1)
    # godmode = np.full_like(source,0)
    # for i in range(n1,source.shape[0]):
        # tci[i] = pine_ema(source,(source - pine_ema(source,source,n1))/(0.025*pine_ema(source,np.abs(source - pine_ema(source,source,n1)),n1)),n2)[i]+50
        # willy[i] = 60 * (source[i] - np.amax(source[i-(n2-1):i+1])) / (np.amax(source[i-(n2-1):i+1]) - np.amin(source[i-(n2-1):i+1])) + 80 
        # u[i] = np.maximum((source[i] - source[i-1]),0)
        # d[i] = np.maximum((source[i-1] - source[i]), 0)
        # sumation1[i] = rsialpha * u[i] + (1 - rsialpha) * (sumation1[i-1])
        # sumation2[i] = rsialpha * d[i] + (1 - rsialpha) * (sumation2[i-1]) 
        # rs[i] = sumation1[i]/sumation2[i]
        # rsi[i] = 100 - 100 / ( 1 + rs[i])
        # pc[i] = source[i] - source[i-1] 
        # double_smoothed_pc = pine_ema(source,pine_ema(source,pc,n1),n2)
        # double_smoothed_pc_abs = pine_ema(source,pine_ema(source,np.abs(pc),n1),n2)
        # tsi_value[i] = (double_smoothed_pc[i] / double_smoothed_pc_abs[i])
        # csi[i] = (rsi[i] + (tsi_value[i]*50+50)) /2 
        # godmode[i] = (csi[i] + willy[i] + tci[i]) / 3 
    # return tci
  
    
# @njit 
# def pine_ema(source1, source2, length):
    # alpha = 2 / (length + 1)
    # sum1 = np.full_like(source1,0)
    # for i in range(0,source1.shape[0]):
        # sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    # return sum1 
      

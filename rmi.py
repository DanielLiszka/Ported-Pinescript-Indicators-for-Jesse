from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

'''
https://www.tradingview.com/script/FXJs9LSi-Relative-Momentum-Index-RMI-Oscillator/
oscillates between 0 and 100. good for finding relative tops and bottoms
''' 
  
def rmi(candles: np.ndarray, length:int=20,momentum:float=4, source_type: str = "close", sequential: bool = False):    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)   
    rmi = rmi_func(source,length,momentum)
    if sequential:
        return rmi
    else:
        return rmi[-1]
        
@njit
def rmi_func(source,length,momentum):
    add_up = np.zeros(source.shape[0])
    minus_down = np.zeros(source.shape[0])
    rmi = np.zeros(source.shape[0])
    add_up_rmi = np.zeros(source.shape[0])
    minus_down_rmi = np.zeros(source.shape[0])
    for i in range(momentum,source.shape[0]):
        add_up[i] = np.maximum((source[i] - source[(i-(momentum))]),0)
        minus_down[i] = abs(np.minimum((source[i] - source[(i-(momentum))]),0))
    add_up_rmi = pine_rma(source,add_up,length)
    minus_down_rmi = pine_rma(source,minus_down,length)
    for i in range(0,source.shape[0]):
        if minus_down_rmi[i] == 0:
            rmi[i] = 100
        elif add_up_rmi[i] == 0:
            rmi[i] = 0
        else:
            rmi[i] = 100 - (100 / (1 + add_up_rmi[i] / minus_down_rmi[i]))
    return rmi
    
@njit 
def pine_rma(source1, source2, length):
    alpha = 1/length
    sum1 = np.zeros(source1.shape[0])
    for i in range(0,source1.shape[0]):
        sum1[i] = alpha * source2[i] + (1 - alpha) * sum1[i-1]
    return sum1 
      
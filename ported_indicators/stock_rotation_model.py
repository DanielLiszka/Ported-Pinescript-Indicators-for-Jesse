from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
import talib

Stock_Rotation_Model = namedtuple("Stock_Rotation_Model",['osc','signal'])

def srm(candles: np.ndarray, length1: int =10, length2: int=25, length3:int=50, length4:int=100,source_type: str = "close", sequential: bool = False) -> Stock_Rotation_Model:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    source = source*2 #temp until source2 is different 
    source2 = source #need to set source2 to larger index like SPY for stocks
    bull1 = talib.ROC(source,length1)
    bull2 = talib.ROC(source,length2)
    bull3 = talib.ROC(source,length3)
    bull4 = talib.ROC(source,length4)
    bull = ((bull1*4) + (bull2 *3) + (bull3 * 2) + bull4)/10
    bear1 = talib.ROC(source2,length1)
    bear2 = talib.ROC(source2,length2)
    bear3 = talib.ROC(source2,length3)
    bear4 = talib.ROC(source2,length4)
    bear = ((bear1 * 4) + (bear2 * 3) + (bear3 * 2) + bear4)/ 10
    srm = talib.WMA((bull - bear), length1)
    slo = srm[-1] - srm[-2] 
    prevslo = srm[-2] - srm[-3]
    if slo > 0:
        if slo > (prevslo):
            signal = 2
        else:
            signal = 1 
    else:
        if slo < (prevslo):
            signal = -2 
        else:  
            signal = -1

    if sequential:
        return Stock_Rotation_Model(srm,signal)
    else:
        return Stock_Rotation_Model(srm[-1],signal)

from typing import Union
import numpy as np
import talib
from numba import njit
from jesse.helpers import get_candle_source
from jesse.helpers import slice_candles
from numpy.lib.stride_tricks import sliding_window_view

#jesse backtest  '2021-01-03' '2021-03-02'
def cmotv(candles: np.ndarray, period: int = 14, source_type: str = "close", sequential: bool = False) -> Union[
    float, np.ndarray]:
    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    f1,f2 = fast_cmo(source,period)
    sm1 = np.sum(sliding_window_view(f1, window_shape=period), axis=1)
    sm2 = np.sum(sliding_window_view(f2, window_shape=period), axis=1)
    chandeMO = 100 * (sm1-sm2)/(sm1+sm2)
    if sequential: 
        return chandeMO
    else:    
        return chandeMO[-1]
    

def fast_cmo(source, period):
    f2 = np.full_like(source,0)
    f1 = np.full_like(source,0)
    mom = np.full_like(source,0)
    chandeMO = np.full_like(source,0)
    for i in range(source.shape[0]):
        mom[i] = (source[i] - source[i-1])
        if mom[i] >= 0:
            f1[i] = mom[i] 
            f2[i] = 0.0
        else:
            f1[i] = 0
            f2[i] = -(mom[i])
    return f1,f2
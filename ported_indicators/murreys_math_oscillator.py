from jesse.helpers import get_candle_source, slice_candles, np_shift
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from collections import namedtuple

#jesse backtest '2021-01-03' '2021-03-02'
"""
no octives used,  https://www.tradingview.com/script/b43I4pmV-RS-UCS-Murrey-s-Math-Oscillator-Modification/#chart-view-comments 
"""

def mm(candles: np.ndarray, length: int= 4, mult:float=0.125, method : str = 'm_donchian',source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:    
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    _hi = talib.MAX(candles[:,3], length)
    _lo = talib.MIN(candles[:,4], length)
    _oscillator,_midline2, _range2 = m_donchian(source,candles,length,mult,_hi,_lo)
    _c = talib.EMA(candles[:,2],length)
    _oscillator2 = 0 - (_c - _midline2) / (_range2 /2)
    if method == 'm_donchian':
        oscillator = _oscillator
    else:
        oscillator = _oscillator2 
    res = _oscillator
    if sequential:
        return oscillator
    else:
        return oscillator[-2:]
@njit    
def m_donchian(source,candles, length, mult, _hi, _lo):
    _range = np.full_like(source,0)
    _range2 = np.full_like(source,0)
    _lvl_range = np.full_like(source,0)
    _lvl_range2 = np.full_like(source,0)
    d_oscillator = np.full_like(source,0)
    _hi2 = np.full_like(source,0)
    _lo2 = np.full_like(source,0)
    _midline = np.full_like(source,0)
    _midline2 = np.full_like(source,0)
    for i in range(length,source.shape[0]):    
        _range[i] = (_hi[i] - _lo[i]) 
        _lvl_range[i] = _range[i] * mult 
        _midline[i] = (_hi[i] + _lo[i])/2
        d_oscillator[i] = (candles[:,2][i] - _midline[i]) / (_range[i] /2)
        _hi2[i] = _hi2[i-1]
        _lo2[i] = _lo2[i-1] 
        if candles[:,3][i] > _hi2[i-1]:
            _hi2[i] = candles[:,3][i]
        else:
            _hi2[i] = np.maximum((((_hi2[i] * (1.0 - mult)) + (candles[:,3][i] * (1.0 + mult)))/2), candles[:,3][i])
        if candles[:,4][i] < _lo2[i-1]:
            _lo2[i] = candles[:,4][i]
        else:  
            _lo2[i] = np.minimum((((_lo2[i] * (1.0 - mult)) + (candles[:,4][i] * (1.0 + mult)))/2), candles[:,4][i]) 
        _range2[i] = _hi2[i] - _lo2[i] 
        _lvl_range2[i] = _range2[i] * mult
        _midline2[i] = (_hi2[i] + _lo2[i])/2
    return d_oscillator, _midline2, _range2 
    
    

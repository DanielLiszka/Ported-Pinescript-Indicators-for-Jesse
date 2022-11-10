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
BollingerBands = namedtuple('Weighted_BollingerBands',['basis', 'upperband', 'lowerband'])

"""
https://www.tradingview.com/script/F6fK5IMa-Function-Weighted-Standard-Deviation/#chart-view-comment-form
"""

def weighted_bollingerbands(candles: np.ndarray, length:int= 20, dev_setting:float=2.0,source_type: str = "close", sequential: bool = False ) -> BollingerBands:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    mean, variance, dev, mse , rmse = fast_bollinger(candles,source,length)
    basis = mean
    upperband = mean + (dev * dev_setting)
    lowerband = mean - (dev * dev_setting)
    if sequential:
        return BollingerBands(basis,upperband,lowerband)
    else:
        return BollingerBands(basis[-1], upperband[-1], lowerband[-1])
    
@njit    
def fast_bollinger(candles,source,length):
    _xw = np.full_like(source,0)
    _sum_weight = np.full_like(source,0)
    _mean = np.full_like(source,0)
    _variance = np.full_like(source,0)
    _dev = np.full_like(source,0)
    _mse = np.full_like(source,0)
    _rmse = np.full_like(source,0)
    for i in range(length+1,source.shape[0]):    
        _xw[i] = source[i] * candles[:,5][i]
        _sum_weight[i] = np.sum(candles[(i-(length-1)):i+1,5])
        _mean[i] = np.sum(_xw[i-(length-1):i+1]) / _sum_weight[i] 
        _sqerror_sum = 0.0
        _nonzero_n = 0.0
        for j in range(length):
            _sqerror_sum = _sqerror_sum + np.power(_mean[i] - source[i-j],2) * candles[:,5][i-j]
            _nonzero_n = _nonzero_n + 1 if candles[:,5][i-j] != 0 else _nonzero_n 
        _variance[i] = _sqerror_sum / ((_nonzero_n - 1) * _sum_weight[i] / _nonzero_n)
        _dev[i] = np.sqrt(_variance[i])
        _mse[i] = _sqerror_sum / _sum_weight[i] 
        _rmse[i] = np.sqrt(_mse[i])
    return _mean, _variance, _dev, _mse, _rmse 
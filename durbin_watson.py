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
from statsmodels.stats.stattools import durbin_watson 
from statsmodels.formula.api import ols
#jesse backtest '2021-01-03' '2021-03-02'

"""
https://www.tradingview.com/script/o8p5hAxr-Durbin-Watson-Test-Statistic-pig/#chart-view-comment-form
WIP
"""

def durbinwatson(candles: np.ndarray, lookback:int=30, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    # model = ols(source[-lookback:],source[-lookback:]).fit()
    # dw = durbin_watson(model.resid,axis=0)

    if sequential:
        return source[-1]
    else:
        return source[-1]
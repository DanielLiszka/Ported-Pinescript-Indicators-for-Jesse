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
from statsmodels.tsa.stattools import adfuller 

#jesse backtest '2021-01-03' '2021-03-02'

ADFULLER = namedtuple('adfuller',['adf', 'pvalue', 'buysignal','strongbuy'])

"""
https://www.tradingview.com/script/KjD8ByIQ-Augmented-Dickey-Fuller-ADF-mean-reversion-test/#chart-view-comment-form
"""

def dickey_fuller(candles: np.ndarray, maxlag:int=0, percentile_type:int=95, length: int= 100, source_type: str = "close", sequential: bool = False ) -> ADFULLER:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    sourceinput = source[-length:]
    adf, pvalue,usedlag,nobs,critical_values,outlier= (adfuller(x=sourceinput, maxlag=maxlag))
    percentile_99, percentile_95, percentile_90 = critical_values['1%'], critical_values['5%'], critical_values['10%']
    if percentile_type == 99:
        percentile = percentile_99
    elif percentile_type == 95:
        percentile = percentile_95
    elif percentile_type == 90:
        percentile = percentile_90
    buysignal = 1 if adf < percentile else 0 
    strongbuy = 1 if adf < percentile and pvalue < 0.05 else 0 
    if sequential:
        return ADFULLER(adf,pvalue,buysignal,strongbuy)
    else:
        return ADFULLER(adf,pvalue,buysignal,strongbuy)
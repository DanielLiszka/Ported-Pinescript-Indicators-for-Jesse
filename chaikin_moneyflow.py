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
tradingview chaikin money flow
"""

def cmf(candles: np.ndarray, length: int= 20, source_type: str = "close", sequential: bool = False ) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    MF = fast_CFH(candles,source,length)
    if sequential:
        return MF
    else:
        return MF[-1]
 
@jit(error_model="numpy")
def fast_CFH(candles,source,length):    
    ad = np.full_like(source,0)
    mf = np.full_like(source,0)
    for i in range(length+1,source.shape[0]):
        ad[i] = (((2*candles[:,2][i]) - candles[:,4][i] - candles[:,3][i])/(candles[:,3][i]-candles[:,4][i]))*candles[:,5][i]
        mf[i] = (np.sum(ad[(i-(length-1)):i+1]) / np.sum(candles[(i-(length-1)):i+1,5]))
    return mf
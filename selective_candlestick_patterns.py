from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
import tulipy as ti
import math 

#jesse backtest  '2021-01-03' '2021-03-02'

"""
25 patterns loaded, threshold is fraction of 50. ex: 14/50
each talib pattern function returns an integer
+200 bullish pattern with confirmation
+100 bullish pattern (most cases)
0 none
-100 bearish pattern
-200 bearish pattern with confirmation
"""
def candlestickpatterns(candles: np.ndarray, source_type: str = "close", threshold:int=0,buythres:int=1, sellthres:int=-1, sequential: bool = False) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential) 
    source = get_candle_source(candles, source_type=source_type) 
    negativeones = np.full_like(source,-1)
    zeros = np.full_like(source,0)
    ones = np.full_like(source,1)
    CDL2CROWS = talib.CDL2CROWS(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDL3BLACKCROWS = talib.CDL3BLACKCROWS(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDL3INSIDE = talib.CDL3INSIDE(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDL3STARSINSOUTH = talib.CDL3STARSINSOUTH(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDL3OUTSIDE = talib.CDL3OUTSIDE(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDL3WHITESOLDIERS = talib.CDL3WHITESOLDIERS(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLADVANCEBLOCK = talib.CDLADVANCEBLOCK(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLABANDONEDBABY = talib.CDLABANDONEDBABY(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDL3LINESTRIKE = talib.CDL3LINESTRIKE(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLBELTHOLD = talib.CDLBELTHOLD(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLBREAKAWAY = talib.CDLBREAKAWAY(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLEVENINGDOJISTAR = talib.CDLEVENINGDOJISTAR(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLEVENINGSTAR = talib.CDLEVENINGSTAR(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLHAMMER = talib.CDLHAMMER(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLHARAMICROSS = talib.CDLHARAMICROSS(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLINVERTEDHAMMER = talib.CDLINVERTEDHAMMER(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLLONGLINE = talib.CDLLONGLINE(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLMATCHINGLOW = talib.CDLMATCHINGLOW(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLMORNINGDOJISTAR = talib.CDLMORNINGDOJISTAR(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLMORNINGSTAR = talib.CDLMORNINGSTAR(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLPIERCING = talib.CDLPIERCING(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLRISEFALL3METHODS = talib.CDLRISEFALL3METHODS(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLSEPARATINGLINES = talib.CDLSEPARATINGLINES(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    CDLTAKURI = talib.CDLTAKURI(candles[:,1],candles[:,3],candles[:,4],candles[:,2])
    score = (CDL2CROWS + CDL3BLACKCROWS + CDL3INSIDE + CDL3STARSINSOUTH + CDL3OUTSIDE + CDL3WHITESOLDIERS + CDLADVANCEBLOCK + CDLABANDONEDBABY \
        + CDL3LINESTRIKE + CDLBELTHOLD + CDLBREAKAWAY + CDLEVENINGDOJISTAR + CDLEVENINGSTAR + CDLHAMMER + CDLHARAMICROSS + CDLINVERTEDHAMMER \
        + CDLLONGLINE + CDLMATCHINGLOW + CDLMORNINGDOJISTAR + CDLMORNINGSTAR + CDLPIERCING + CDLRISEFALL3METHODS + CDLSEPARATINGLINES + CDLTAKURI) / 100
    scorethreshold = ones if np.abs(score[-1]) > threshold else zeros 
    if scorethreshold[-1] == 1 and score[-1] >= buythres:
        signal = ones 
    elif scorethreshold[-1] == 1 and score[-1] <= sellthres:
        signal = negativeones 
    else:
        signal = zeros 
    if sequential:
        return signal
    else:
        return signal[-1]

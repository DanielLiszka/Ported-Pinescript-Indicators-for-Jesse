from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple

JMA_MACD = namedtuple('JMA_MACD',['Jurik_MACD_Leader', 'Jurik_MACD', 'Signal_Line','Histogram'])

"""
https://www.tradingview.com/script/ajZov02O-Jurik-MacD-Leader-NCM/#chart-view-comment-form
"""

def jma_macd(candles: np.ndarray, fast_period: int = 7, fast_phase: int = -100, fast_power:float=2, signal_period: int = 9, signal_phase:int = 50, signal_power: float=2,slow_period: int= 40,slow_phase: int = 0, slow_power: float= 1,  source_type: str = "close", sequential: bool = False) -> JMA_MACD:
    candles = candles[-800:] #slice_candles(candles, sequential)
    source = candles[:,2] #get_candle_source(candles, source_type=source_type)
    slow_jma = jma_fast(source, slow_period, slow_phase, slow_power)
    fast_jma = jma_fast(source,fast_period,fast_phase,fast_power)
    jmacd = fast_jma - slow_jma
    signal_jma = jma_fast(jmacd, signal_period, signal_phase,signal_power)
    hist = jmacd - signal_jma
    source1 = source - fast_jma
    source2 = source - slow_jma
    indicator1 = fast_jma + jma_fast(source1, fast_period, slow_phase, slow_power)
    indicator2 = slow_jma + jma_fast(source2, slow_period, slow_phase, slow_power)
    jmacdLeader = indicator1 - indicator2
    
    if sequential:
        return JMA_MACD(jmacdLeader,jmacd,signal_jma,hist)
    else:
        return JMA_MACD(jmacdLeader[-1],jmacd[-1],signal_jma[-1],hist[-1])


@njit(fastmath=True) 
def jma_fast(source, period, phase, power):
    e0 = np.zeros(source.shape[0])
    e1 = np.zeros(source.shape[0])
    e2 = np.zeros(source.shape[0])
    jma = np.zeros(source.shape[0])
    phaseRatio = 0.0
    beta = 0.45 * (period - 1) / (0.45 * (period -1) + 2) 
    alpha = np.power(beta,power)
    if phase < -100: 
        phaseRatio = 0.5
    elif phase > 100: 
        phaseRatio = 2.5
    else: 
        phaseRatio = phase / 100 + 1.5
    for i in range(source.shape[0]):
        e0[i] = (1 - alpha) * source[i] + alpha * e0[i-1]
        e1[i] = (source[i] - e0[i]) * (1 - beta) + beta * e1[i-1] 
        e2[i] = (e0[i] + phaseRatio * e1[i] - jma[i-1]) * np.power(1 - alpha, 2) + np.power(alpha, 2) * e2[i-1] 
        jma[i] = e2[i] + jma[i-1]
    return jma
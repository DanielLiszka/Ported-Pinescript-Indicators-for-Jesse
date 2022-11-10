from jesse.helpers import get_candle_source, slice_candles
import numpy as np
from numba import njit, jit
import talib 
from typing import Union
from jesse.helpers import get_config

"""
https://www.tradingview.com/script/a7062wqw-Adaptive-Trend-Cipher-loxx/
"""

def adaptive_length(candles: np.ndarray,b_corr:bool=False,smooth:int = 8,bp_period: int =13, bandwidth:float=0.20, LPPeriod: int = 20,hilbert_len:int=7, alpha : float = 0.07,adaptive_alg: str = "band pass", smoothing: bool = False, source_type: str = "hl2", sequential: bool = False) -> Union[float, np.ndarray]:
    candles1 = slice_candles(candles, sequential)
    source = get_candle_source(candles1, source_type=source_type)
    if adaptive_alg == "icp":
        f_ICP, smoothICP = func_icp(source,smooth)
        if smoothing == True:
            adaptive = smoothICP
        else:
            adaptive = f_ICP
    elif adaptive_alg == "hilbert_dual":
        hilbert_dual = func_hilbert_dual(source,LPPeriod)
        adaptive = hilbert_dual
    elif adaptive_alg == "hilbert":
        hilbert = func_hilbert(source,hilbert_len,alpha)
        adaptive = hilbert
    elif adaptive_alg == "band pass":
        bandpass = func_bandpass(source,bp_period,bandwidth)
        adaptive = bandpass
    else:
        print("value error")
    # elif 
    if b_corr:
        corr = correlation(source,candles,int(adaptive[-1]))
    else:
        corr = np.zeros(2)
    if sequential:
        return adaptive,corr
    else:
        return adaptive[-1],corr[-1]

@njit
def func_icp(source,smooth):
    RADIANtoDEGREES = 90.0 / np.arcsin(1.0)
    SQRT2xPI = np.sqrt(2.0) * np.arcsin(1.0) * 2.0
    detrend = np.zeros(source.shape[0])
    inPhase = np.zeros(source.shape[0])
    quadrature = np.zeros(source.shape[0])
    divisor = np.zeros(source.shape[0])
    phase = np.zeros(source.shape[0])
    deltaPhase = np.zeros(source.shape[0])
    smoothICP = np.zeros(source.shape[0])
    f_ICP = np.zeros(source.shape[0])
    alpha = SQRT2xPI / smooth 
    beta = np.exp(-alpha)
    gamma = -beta* beta
    delta = 2.0 * beta * np.cos(alpha)
    for i in range(81,source.shape[0]):
        detrend[i] = source[i] - source[i-7]
        inPhase[i] = 1.25 * (detrend[i-4] - 0.635 * detrend[i-2]) + 0.635 * inPhase[i-3]
        quadrature[i] = detrend[i-2] - 0.338 * detrend[i] + 0.338 * quadrature[i-2]
        divisor[i] = inPhase[i] + inPhase[i-1]
        phase[i] = RADIANtoDEGREES * np.arctan(np.abs((quadrature[i] + quadrature[i-1])/divisor[i])) if divisor[i] != 0.0 else 0.0
        if (inPhase[i] < 0.0 and quadrature[i] > 0.0):
            phase[i] = 180.0 - phase[i]
        if (inPhase[i] < 0.0 and quadrature[i] < 0.0):
            phase[i] = 180.0 + phase[i]
        if (inPhase[i] > 0.0 and quadrature[i] < 0.0):
            phase[i] = 360 - phase[i] 
        deltaPhase[i] = phase[i-1] - phase[i] 
        if (phase[i-1] < 90 and phase[i] > 270):
            deltaPhase[i] = 360 + phase[i-1] - phase[i] 
        deltaPhase[i] = np.maximum(1.0, np.minimum(60.0,deltaPhase[i]))
        E = 0.0
        ICP = 0.0
        for j in range(81):
            E = E + deltaPhase[i-j]
            if (E > 360.0 and ICP == 0):
                ICP = (j)
                break 
        f_ICP[i] = ICP
        if (f_ICP[i]==0):
            f_ICP[i] = f_ICP[i-1]
        smoothICP[i] = (1.0 - delta - gamma) * f_ICP[i] + delta * smoothICP[i-1] + gamma * smoothICP[i-2]
        f_ICP[i] = np.maximum(6,f_ICP[i])
        smoothICP[i] = np.maximum(6,smoothICP[i])
    return detrend, np.floor(smoothICP)

@njit(fastmath = True)
def func_hilbert_dual(source, LPPeriod): 
    max_len = 80
    min_len = 6
    alpha = (np.cos(0.707 * 2 * np.pi / max_len) + np.sin(0.707 * 2 * np.pi / max_len) - 1) / np.cos(0.707 * 2 * np.pi / max_len)
    a1 = np.exp(-np.sqrt(2) * np.pi / LPPeriod)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / LPPeriod)
    c2 = b1 
    c3 = -a1 * a1 
    c1 = 1 - c2 - c3 
    HP = np.zeros(source.shape[0])
    Filt = np.zeros(source.shape[0])
    IPeak = np.zeros(source.shape[0])
    Real = np.zeros(source.shape[0])
    Quad = np.zeros(source.shape[0])
    QPeak = np.zeros(source.shape[0])
    Imag = np.zeros(source.shape[0])
    IDot = np.zeros(source.shape[0])
    QDot = np.zeros(source.shape[0])
    Period = np.zeros(source.shape[0])
    DomCycle = np.zeros(source.shape[0])
    for i in range(source.shape[0]):
        HP[i] = (1 - alpha / 2) * (1 - alpha / 2) * (source[i] - 2 * source[i-1] + source[i-2]) + 2 * (1 - alpha) * HP[i-1] - (1 - alpha) * (1 - alpha) * HP[i-2]
        Filt[i] = c1 * (HP[i] + HP[i-1]) / 2 + c2 * Filt[i-1] + c3 * Filt[i-2] 
        IPeak[i] = 0.991 * IPeak[i-1] 
        IPeak[i] = np.abs(Filt[i]) if np.abs(Filt[i]) > IPeak[i] else IPeak[i]
        Real[i] = Filt[i] / IPeak[i] 
        Quad[i] = Real[i] - Real[i-1] 
        QPeak[i] = 0.991 * QPeak[i-1] 
        QPeak[i] = np.abs(Quad[i]) if np.abs(Quad[i]) > QPeak[i] else QPeak[i]
        Imag[i] = Quad[i] / QPeak[i] 
        IDot[i] = Real[i] - Real[i-1] 
        QDot[i] = Imag[i] - Imag[i-1] 
        Period[i] = 2 * np.pi * (Real[i] * Real[i] + Imag[i] * Imag[i]) / (-Real[i] * QDot[i] + Imag[i] * IDot[i]) if Real[i] * QDot[i] - Imag[i] * IDot[i] != 0 else Period[i-1] 
        Period[i] = np.minimum(np.maximum(Period[i],min_len),max_len)
        DomCycle[i] = c1 * (Period[i] + Period[i-1]) / 2 + c2 * DomCycle[i-1] + c3 * DomCycle[i-2] 
    return np.floor(DomCycle)

#might not be accurate
@njit(fastmath=True)
def correlation(x,candles,len1):
    min_len = 6
    max_len = 80
    meanx = np.zeros(x.shape[0])
    meany = np.zeros(x.shape[0]) 
    output = np.zeros(x.shape[0])
    y = np.zeros(x.shape[0])
    for i in range(len1,x.shape[0]):
        y[i] = candles.shape[0] - i
        meanx[i] = np.mean(x[i-(len1-1):i+1])
        meany[i] = np.mean(y[i-(len1-1):i+1])
        sumx = 0.0
        sumy = 0.0
        sumxy = 0.0
        for j in range(max_len):
            sumxy = (sumxy +x[i-j] - meanx[i]) * y[i-j] - meany[i]
            sumx = sumx + np.power(x[i-j] - meanx[i],2)
            sumy = sumy + np.power(y[i-j] - meany[i], 2)
        output[i] = sumxy / np.sqrt(sumy * sumx)
    return output

#not accurate; np.percentile replaced with cheap alternative 
@njit(fastmath=True)
def func_hilbert(source,hilbert_len,alpha):
    smooth = np.zeros(source.shape[0])
    cycle  = np.zeros(source.shape[0])
    period = np.zeros(source.shape[0])
    q1  = np.zeros(source.shape[0])
    deltaPhase  = np.zeros(source.shape[0])
    medianDelta  = np.zeros(source.shape[0])
    dc  = np.zeros(source.shape[0])
    instPeriod  = np.zeros(source.shape[0])
    WMA  = np.zeros(source.shape[0])
    index = (np.int(np.floor(hilbert_len/2)))
    for i in range(hilbert_len,source.shape[0]):
        smooth[i] = (source[i] + 2 * source[i-1] + 2 * source[i-2] + source[i-3]) / 6 
        cycle[i] = (1- 0.5 * alpha) * (1 - 0.5 * alpha) * (smooth[i] - 2 * smooth[i-1] + smooth[i-2]) + 2 * (1 - alpha) * cycle[i-1] - (1 - alpha) * (1 - alpha) * cycle[i-2]
        q1[i] = (0.0962 * cycle[i] + 0.5769 * cycle[i-2] - 0.5769 * cycle[i-4] - 0.0962 * cycle[i-6]) * (0.5 + 0.08 * instPeriod[i-1])
        deltaPhase[i] = (cycle[i-3] / q1[i] - cycle[i-4] / q1[i-1]) / (1 + cycle[i-3] * cycle[i-4] / (q1[i] * q1[i-1])) if q1[i] != 0 and q1[i-1] != 0 else 0 
        deltaPhase[i] = np.minimum(np.maximum(deltaPhase[i],0.1),1.1)
        medianDelta[i] = deltaPhase[i-index] #np.percentile(deltaPhase[i-(hilbert_len-1):i+1],50)
        dc[i] = np.pi * 2 / medianDelta[i] + 0.5 if medianDelta[i] != 0 else 15
        instPeriod[i] = 0.33 * dc[i] + 0.67 * (instPeriod[i-1])
        period[i] = 0.15 * instPeriod[i] + 0.85 * period[i-1] 
        weight = 0.0
        norm = 0.0 
        sum1 = 0.0
        for j in range(4):
            weight = (4 - j)*4
            norm = norm + weight 
            sum1 = sum1 + period[i-j] * weight  
        WMA[i] = np.floor(sum1/norm)
    return np.floor(WMA) 
    
#not accurate
@njit(fastmath=True)
def func_bandpass(source,bp_period,bpw):
    alpha2 = (np.cos(0.25*bpw*2*np.pi/bp_period) + np.sin(0.25 * bpw * 2 * np.pi / bp_period) - 1) / np.cos(0.25 * bpw * 2 * np.pi / bp_period)
    beta1 = (np.cos(2*np.pi/bp_period))
    gamma1 = 1 / np.cos(2 * np.pi * bpw / bp_period)
    alpha1 = gamma1 - np.sqrt(gamma1 * gamma1 - 1)
    HP = np.zeros(source.shape[0])
    BP = np.zeros(source.shape[0])
    Peak = np.zeros(source.shape[0])
    Real = np.zeros(source.shape[0])
    DC = np.zeros(source.shape[0])
    counter = np.zeros(source.shape[0])
    for i in range(bp_period,source.shape[0]):
        HP[i] = (1 + alpha2 / 2) * (source[i] - source[i-1]) + (1 - alpha2) * HP[i-1]
        BP[i] = 0.5 * (1 - alpha1) * (HP[i] - HP[i-2]) + beta1 * (1 + alpha1) * BP[i-1] - alpha1 * BP[i-2]
        Peak[i] = 0.991 * Peak[i-1]
        Peak[i] = np.abs(BP[i]) if np.abs(BP[i]) > Peak[i] else Peak[i] 
        Real[i] = BP[i] / Peak[i] if Peak[i] != 0 else Real[i-1]
        DC[i] = 6 if DC[i-1] < 6 else DC[i-1] 
        counter[i] = counter[i-1] + 1 
        if (Real[i] > 0 and Real[i-1] < 0) or (Real[i] < 0 and Real[i-1] > 0):
            DC[i] = 2 * counter[i] 
            if 2 * counter[i] > 1.25 * DC[i-1]:
                DC[i] = 1.25 * DC[i-1] 
                if 2 * counter[i] > 1.25 * DC[i-1]:
                    DC[i] = 1.25 * DC[i-1] 
                if 2 * counter[i] < 0.8 * DC[i-1]: 
                    DC[i] = 0.8 * DC[i-1] 
                counter[i] = 0 
    return np.floor(DC) 
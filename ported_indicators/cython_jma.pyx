#cython: wraparound = True
#cython: boundscheck = False
#cython: nonecheck = False
#cython: cdivision = False
from jesse.helpers import get_candle_source, slice_candles
import numpy as np
cimport numpy as np 
# from cython cimport view
# from libc.stdlib cimport malloc, free
cimport cython
np.import_array()



def c_jma(double [:,::1] candles , period: int= 100,phase: int = 50, power: float= 2,  source_type: str = "close", sequential: bool = False):
    candles = candles[-800:] #slice_candles(candles, sequential)
    cdef double [:] source = candles[:,2] #get_candle_source(candles, source_type=source_type)
    jma = jma_fast(source, period, phase, power)
    # jma = np.asarray(jma)
    if sequential:
        return jma
    else:
        return jma
        
        
#jesse backtest  '2021-01-03' '2021-03-02' --json

cdef double[::1] jma_fast(double [:] source, int period, int phase, float power):
    cdef double[::1] e0, e1, e2, jma
    cdef double phaseRatio, beta, alpha
    cdef Py_ssize_t shape1 = source.shape[0] 
    # cdef int signal 
    cdef Py_ssize_t i 
    # cdef double *e0 = <double *>malloc(shape1 * sizeof(double))
    # cdef double *e1 = <double *>malloc(shape1 * sizeof(double))
    # cdef double *e2 = <double *>malloc(shape1 * sizeof(double))
    # cdef double *jma = <double *>malloc(shape1 * sizeof(double))
    e0 = np.empty(shape1)
    e1 = np.empty(shape1)
    e2 = np.empty(shape1)
    jma = np.empty(shape1)
    with nogil:
        for i in range(shape1):
            e0[i] = 0 
            e1[i] = 0
            e2[i] = 0 
            jma[i] = 0

        phaseRatio = 0.0
        beta = 0.45 * (period - 1) / (0.45 * (period -1) + 2) 
        alpha = beta ** power
        if phase < -100: 
            phaseRatio = 0.5
        elif phase > 100: 
            phaseRatio = 2.5
        else: 
            phaseRatio = phase / 100 + 1.5
        for i in range(0,shape1):
            e0[i] = (1 - alpha) * source[i] + alpha * e0[i-1]
            e1[i] = (source[i] - e0[i]) * (1 - beta) + beta * e1[i-1] 
            e2[i] = (e0[i] + phaseRatio * e1[i] - jma[i-1]) * (1 - alpha)**2 + alpha**2 * e2[i-1] 
            jma[i] = e2[i] + jma[i-1]
        # signal = jma[-1]
        # free(e0)
        # free(e1)
        # free(e2)
        # free(jma)
        return jma[-2:]
    
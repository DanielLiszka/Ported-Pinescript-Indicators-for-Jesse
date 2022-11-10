from jesse.helpers import get_candle_source
import numpy as np
# from numba import njit
# import talib 
from math import exp
from typing import Union
from jesse.helpers import get_config
from collections import namedtuple
from talipp.indicators import ALMA 
import _pickle as cPickle
import gc
import flatbuffers
#jesse backtest  '2021-01-03' '2021-03-02'  
'''
talipp test Alma, needs more work to become incremental, offset rounds down to nearest tenths place   
''' 
  
def t_alma(candles: np.ndarray, length:int=11, offset:float=0.8, sigma:float=6, source_type: str = "close", sequential: bool = False, firstrun:bool=False) -> Union[float, np.ndarray]:    
    candles = candles[-480:] #slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)   
    # alma = ALMA(length, offset, sigma, source)
    global alma, alma_pickled 
    if firstrun == True:
        alma = ALMA(length, offset, sigma, source)
        alma.purge_oldest(len(source)-(length))
        alma_pickled = cPickle.dumps(alma,protocol=-1)
    else:
        gc.disable()
        prealma = cPickle.loads(alma_pickled)
        prealma.add_input_value(source[-1])
        prealma.purge_oldest(1)
        alma = np.asarray(prealma)
        alma_pickled = cPickle.dumps(prealma,protocol=-1)
        gc.enable()
    if sequential:
        return alma[-1]
    else:
        return alma[-1]



"""        
from typing import List
from math import exp


def alma(source,length,offset,sigma,firstrun,old_input_values,new_input_values)
        # calculate weights and normalisation factor (w_sum)
        if new_input_values is not None:
            old_input_values.append(new_input_values)
        w = []
        w_sum = 0.0
        s = period / float(sigma)
        m = int((period - 1) * offset)
        for i in range(0, period):
            w.append(exp(-1 * (i - m) * (i - m) / (2 * s * s)))
            w_sum += w[-1]
        if firstrun == True:
            return w_sum
        else:
            alma = 0.0
            for i in range(0,period):
                alma += old_input_values[-(period - i)] * w[i]
            return alma / w_sum
"""

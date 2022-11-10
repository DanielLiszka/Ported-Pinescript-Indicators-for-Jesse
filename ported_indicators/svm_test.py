
from sklearn import mixture as mix
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from jesse.helpers import get_candle_source, slice_candles, np_shift, same_length
import numpy as np
from numba import njit,jit
from typing import Union
from jesse.helpers import get_config
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pandas as pd 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

# def SVM( Data
    # X_Train = Data 
    # sc = StandardScalar()
    # X_Train = sc.fit_transform(X_train)
    # Y_Train = Candles[:,0] 
    
#jesse backtest  '2021-01-03' '2021-03-02'
    
def NN( candles: np.ndarray, sequential: bool = True, source_type: str = "close",) -> Union[float, np.ndarray]:
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)  
    y_train = np.full_like(source,0)
    sc = StandardScaler()
    Data = candles
    X_train = sc.fit_transform(Data)
    X_test = sc.transform(X_train)
    classifier = Sequential()
    classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = candles.shape[1]))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 1, epochs = 100)
    y_pred = classifier.predict(X_test)
    y_pred = 1 if (y_pred[-1] > 0.5) else -1 
    if sequential: 
        return y_pred[-1]
    else:    
        return y_pred[-1]
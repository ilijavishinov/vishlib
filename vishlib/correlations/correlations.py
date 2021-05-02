import numpy as np
import pandas as pd
from typing import List, Union

def pearson_linear_correlation(x: Union[pd.Series, np.ndarray, List],
                               y: Union[pd.Series, np.ndarray, List]):
    """
    
    :param x:
    :param y:
    :return:
    """
    
    # convertions
    if x is pd.Series: x = x.to_numpy()
    if x is List: x = np.ndarray(x)
    if y is pd.Series: y = y.to_numpy()
    if y is List: y = np.ndarray(y)
    
    # assertions
    assert len(x) == len(y), 'Arrays must be of same length!'
    
    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x), np.std(y)
    
    covariance_tmp = 0
    for x_i, y_i in zip(x, y):
        covariance_tmp += (x_i - mean_x) * (y_i - mean_y)
    covariance = covariance_tmp / len(x)
    
    r = covariance / (std_x * std_y)
    pval = None

    return r, pval
    
    
    
    
    
    
    
    
    
    
    
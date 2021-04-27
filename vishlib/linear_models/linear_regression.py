import numpy as np


class LinearRegression:
    
    
    def __init__(self, max_iter):
        
        # constructor args
        self.max_iter = max_iter
        
        
    
    #-----------------------------------------------------------------
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        
        # initialize weights
        weights = np.zeros(X_train.shape[1])
        num_instances = X_train.shape[0]
        learning_rate = 0.05
        
        # perform gradient descent
        for iter_pos in range(self.max_iter):
            
            # make predictions and calculate loss on train set
            y_pred = np.multiply(X_train, weights).sum(axis = 1)
            loss = np.sum((y_train - y_pred)**2)/(2*num_instances)
            
            # update weights
            for weight_idx in range(weights.size):
                # weights[weight_idx] =
                pass
            pass
        
                
                
                
                
                
                
            
            
        
        
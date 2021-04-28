import numpy as np

class LinearRegression:
    
    def __init__(self, max_iter):
        # constructor args
        self.max_iter = max_iter

    # -----------------------------------------------------------------
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # initialize weights
        weights = np.zeros(X_train.shape[1])
        num_instances = X_train.shape[0]
        learning_rate = 0.05
        
        # perform gradient descent
        for iter_pos in range(self.max_iter):
            
            # make predictions and calculate loss on train set
            y_pred = np.multiply(X_train, weights).sum(axis = 1)
            loss = np.sum((y_train - y_pred) ** 2) / (2 * num_instances)
            
            # update weights
            updatad_weights = np.array([np.nan] * weights.size)
            for weight_idx in range(weights.size):
                
                # calculate update component (partial derivative mul. by learning rate)
                partial_derivative = 0
                for row_i, y_i, y_pred_i in zip(X_train, y_train, y_pred):
                    partial_derivative += (y_pred_i - y_i) * row_i[weight_idx]
                partial_derivative /= num_instances
                update_component = partial_derivative * learning_rate
                
                updatad_weights[weight_idx] = weights[weight_idx] - update_component
            weights = updatad_weights
                
            
                
                
                
                
            
            
        
        
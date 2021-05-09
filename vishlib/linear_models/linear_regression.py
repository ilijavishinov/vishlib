import numpy as np
from typing import Union
from vishlib_abstract_classes import Estimator

class LinearRegression(Estimator):
    
    def __init__(self,
                 method: str = 'analytical',
                 warm_start: bool = False,
                 include_bias: bool = False,
                 max_iter: int = 100,
                 batch_size: Union[int, float] = None,
                 learning_rate: float = 1e-2
                 ):
        
        # constructor args
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.include_bias = include_bias
        self.warm_start = warm_start
        self.method = method

        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        
        if self.method == 'gd':
            self.fit_gd(X_train, y_train)
        if self.method == 'analytical':
            self.fit_analytical(X_train, y_train)
            
    
    def fit_analytical(self, X_train: np.ndarray, y_train: np.ndarray):
    
        num_instances = X_train.shape[0]
        if self.include_bias:
            X_train = np.concatenate(X_train, np.ones(num_instances).reshape((num_instances, 1)), axis = 1)

        weights =\
            np.dot(
                np.linalg.inv(
                    np.matmul(
                        np.transpose(X_train),
                        X_train
                    )
                ),
                np.dot(
                    np.transpose(X_train),
                    y_train
                )
            )

        if self.include_bias:
            self.bias = weights[-1]
            self.weights = weights[:-1]
        else:
            self.weights = weights
            self.bias = 0

    
    def fit_gd(self, X_train: np.ndarray, y_train: np.ndarray):
        
        # initialize weights
        weights = np.zeros(X_train.shape[1])
        bias = 0
        num_instances = X_train.shape[0]
        learning_rate = self.learning_rate

        # calculate batch splitting
        if self.batch_size is None:
            num_batches = 1
        else:
            num_batches = num_instances // self.batch_size
        
        # perform gradient descent
        for iter_pos in range(self.max_iter):
            
            for X_train_batch, y_train_batch in zip(np.array_split(X_train, num_batches, axis = 0),
                                                    np.array_split(y_train, num_batches, axis = 0)):
                
                # make predictions and calculate loss on train set
                y_pred = np.multiply(X_train_batch, weights).sum(axis = 1) + bias
                loss = np.sum((y_train_batch - y_pred) ** 2) / (2 * num_instances)
                
                # update weights
                updatad_weights = np.array([np.nan] * weights.size)
                for weight_idx in range(weights.size):
                    # calculate update component (partial derivative mul. by learning rate)
                    partial_derivative = 0
                    for row_i, y_i, y_pred_i in zip(X_train_batch, y_train_batch, y_pred):
                        partial_derivative += (y_pred_i - y_i) * row_i[weight_idx]
                    partial_derivative /= num_instances
                    update_component = partial_derivative * learning_rate
                    updatad_weights[weight_idx] = weights[weight_idx] - update_component
                weights = updatad_weights
                
                # update bias if included
                if self.include_bias:
                    partial_derivative = 0
                    for y_i, y_pred_i in zip(y_train_batch, y_pred):
                        partial_derivative += (y_pred_i - y_i)
                    partial_derivative /= num_instances
                    update_component = partial_derivative * learning_rate
                    bias -= update_component
                    
        self.weights = weights
        self.bias = bias


    def predict(self, X_test: np.ndarray, y_test: np.ndarray):
        return np.multiply(X_test, self.weights).sum(axis = 1) + self.bias
    
    
                
                
            
                
                
                
                
            
            
        
        
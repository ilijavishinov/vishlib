import numpy as np
from typing import Union
from vishlib_abstract_classes import Estimator

class LogisticRegression(Estimator):
    
    def __init__(self,
                 method: str = 'gd',
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
        
    
    def sigmoid(self, input):
        return 1 / (1 + math.e ** (-input))
    
    
    # def fit(self, X_train: np.ndarray, y_train: np.ndarray):
    #
    #     # initialize weights
    #     weights = np.zeros(X_train.shape[1])
    #     bias = 0
    #     num_instances = X_train.shape[0]
    #     learning_rate = self.learning_rate
    #
    #     # calculate batch splitting
    #     if self.batch_size is None:
    #         num_batches = 1
    #     else:
    #         num_batches = num_instances // self.batch_size
    #
    #     # perform gradient descent
    #     for iter_pos in range(self.max_iter):
    #         for X_train_batch, y_train_batch in zip(np.array_split(X_train, num_batches, axis = 0),
    #                                                 np.array_split(y_train, num_batches, axis = 0)):
    #             # make predictions and calculate loss on train set
    #             y_pred_probas = self.sigmoid(np.multiply(X_train_batch, weights).sum(axis = 1) + bias)
                
                
        # TO BE CONTINUED
    
    def predict_probas(self, X_test):
        return self.sigmoid(np.multiply(X_test, weights).sum(axis = 1) + bias)

    
    def predict(self, X_test):
        np.where(self.predict_probas(X_test) > 0.5, 1, 0)
        











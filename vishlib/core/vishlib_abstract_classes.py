from abc import ABC

class Estimator(ABC):
    
    def fit(self):
        pass
    
    def predict(self):
        pass
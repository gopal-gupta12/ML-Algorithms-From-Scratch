import numpy as np
from collections import Counter

def EuclideanDistance(x1, x2):
     return np.sqrt(np.sum((x1 - x2)**2))
     
class KNN:

    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X , y):
         self.X_train = X
         self.y_train = y

    def predict(self , X):
         predictions = [self.helperpredict(x) for x in X]
         return predictions
    
    def helperpredict(self, x):
         distances = [EuclideanDistance(x, x_train) for x_train in self.X_train] 

         k_idxs = np.argsort(distances)[:self.k] 
         k_nearestLabel = [self.y_train[i] for i in k_idxs]

         most_common_label =  Counter(k_nearestLabel).most_common()
         return most_common_label[0][0]
    

from NaiveBayes import NaiveBayes 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import datasets
from sklearn.model_selection import cross_val_score

X , y = datasets.make_classification(n_samples= 1000 , n_features= 10 , n_classes= 2 , random_state = 123)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state= 123)

model = NaiveBayes()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Accuracy Score :
def accuracy_score(pred , y_test):
    acc = np.sum(pred == y_test) / len(y_test)   
    return acc

print("Accuracy Score ", accuracy_score(pred , y_test))


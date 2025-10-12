from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest

data = load_breast_cancer()
X , y = data.data , data.target

X_train , X_test , y_train , y_test = train_test_split(X , y, test_size= 0.2)
model = RandomForest()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Accuracy Score :
def accuracy_score(pred , y_test):
    acc = np.sum(pred == y_test) / len(y_test)   
    return acc

print(f"Accuracy : {round(accuracy_score(pred, y_test),2)} ")
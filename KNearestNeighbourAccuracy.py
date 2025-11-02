from KNN import KNN
from sklearn.datasets import load_iris 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_iris()
X , y = data.data , data.target

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size= 0.2 , random_state= 123)

model = KNN(k = 4)
model.fit(X_train , y_train)
pred = model.predict(X_test)

print("Accuracy Score :", round(accuracy_score(pred , y_test), 2))
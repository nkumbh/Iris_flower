[iris_flower.csv](https://github.com/user-attachments/files/17495777/iris_flower.csv)
import pandas as pd
import numpy as np

data = pd.read_csv('iris_flower.csv')

data.head()
data.drop(columns = ['Id'])
data.info()
# data['SepalLengthCm'].hist()
# data['SepalWidthCm'].hist()       #Blue - SepalLenth Orange- SepalWidth
# data['PetalLengthCm'].hist()
# data['PetalWidthCm'].hist()  #Blue - PetalLength Orange- PetalWidth
X = data.drop(['Id','Species'], axis=1)
y = data['Species']
print(X.head())
print(X.shape)
print(y.head())
print(y.shape)
import pandas as pd

Y = y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

print(y.unique())
print(Y.unique())
Y = Y.values
print(Y)
from sklearn.preprocessing import StandardScaler
data.describe().loc[['min','max']].T
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dt = X.iloc[:,:]
scaled_data = scaler.fit_transform(dt.values)
print(scaled_data)
X = scaled_data
y = Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, epochs=20):
   
    n_samples, n_features = X.shape
    weights = np.zeros(n_features + 1)  

    for epoch in range(epochs):
        z = np.dot(X, weights[1:]) + weights[0]
        p = sigmoid(z)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        # Gradient descent update
        weights[1:] -= learning_rate * np.dot(X.T, (p - y)) / n_samples
        
        weights[0] -= learning_rate * np.mean(p - y)  
        
    def predict(X):
        z = np.dot(X, weights[1:]) + weights[0]
        p = sigmoid(z)
        return np.round(p)

    return weights, predict   
    
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

print(X_test.shape)
print(y_test.shape)
print(X_train.shape)
print(y_train.shape)
weights, predict = logistic_regression(X_train, y_train)

y_pred = predict(X_test)

y_pred = predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the logistic regression model: {accuracy * 100:.4f}%")
y_pred = predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy of the logistic regression model: {accuracy * 100:.4f}%")

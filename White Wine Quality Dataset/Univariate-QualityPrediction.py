# ML models implemented here uses only one parameter 'alcohol' from the dataset to predict the 'quality' of Wine.
# Training Subset = 80% and Testing Subset = 20%

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import mean_squared_error as mse
from sklearn import linear_model
from sklearn import neighbors

def get_tts_from_df(dataFrame, test_size, attrs, labels):
    train, test = tts(dataFrame, test_size=test_size)
    X_train = (train[attrs]).to_numpy().reshape(-1,1)
    Y_train = (train[labels]).to_numpy().reshape(-1,1)
    X_test = (test[attrs]).to_numpy().reshape(-1,1)
    Y_test = (test[labels]).to_numpy().reshape(-1,1)
    return (X_train, X_test, Y_train, Y_test)
    
whiteWine = pd.read_csv("whitewine-quality.csv", sep=",", usecols=['alcohol', 'quality'])
labels = ['quality']
attrs = whiteWine.drop(labels, axis=1).columns.values.tolist()
X_train, X_test, Y_train, Y_test = get_tts_from_df(whiteWine, 0.2, attrs, labels)

whiteWine.plot(kind='scatter', x='alcohol', y='quality', color='red')
plt.show()

#Linear Regression
lin_reg_model = linear_model.LinearRegression()
lin_reg_model.fit(X_train, Y_train)

Y_pred = lin_reg_model.predict(X_test)
print("Mean Sqaured Error: ", round(mse(Y_test, Y_pred),3))

#k-Nearest Neighbors
knn_model = neighbors.KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, Y_train)

Y_pred = knn_model.predict(X_test)
print("Mean Sqaured Error: ", round(mse(Y_test, Y_pred),3))
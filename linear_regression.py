import pandas as pd
import numpy as np

df = pd.read_csv('./data/food_expenditure.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# train LR model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# predict on test set
y_pred = lr.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color='green')
plt.scatter(X_test, y_test, color='red')
plt.plot(X, lr.predict(X), color='black')
plt.show()
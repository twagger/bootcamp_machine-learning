"""Linear model module"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../ex03/')
from my_linear_regression import MyLinearRegression


if __name__ == '__main__':

    # load dataframe
    df = pd.read_csv('./are_blue_pills_magics.csv')

    # perform a linear regression
    mlr = MyLinearRegression(np.array([[0.0], [0.0]]), max_iter=150000)
    x = np.array(df['Micrograms'])
    y = np.array(df['Score'])

    # 1. predict
    y_hat = mlr.predict_(x)
    print(f'First prediction with thetas [[0.0], [0.0]] :\n{y_hat}\n')

    # 2. evaluate : estimate loss
    loss = mlr.loss_(y, y_hat)
    print(f'Loss for first prediction: {loss}\n')

    # 3. improve : fit model
    thetas = mlr.fit_(x, y)
    print(f'Thetas after fitting :\n{thetas}\n')
    y_hat = mlr.predict_(x)
    print(f'Prediction after fitting :\n{y_hat}\n')
    print(f'Loss after fitting : {mlr.loss_(y, y_hat)}\n')

    # 4. plotting the final result
    plt.figure()
    plt.scatter(x, y, marker='o')
    plt.plot(x, y_hat, color='green', ls='--')
    plt.grid(visible=True)
    plt.xlabel('Quantity of blue pills (in migrograms)')
    plt.ylabel('Space driving score')
    plt.show()


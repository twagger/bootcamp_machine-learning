"""Linear model module"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../ex03/')
from my_linear_regression import MyLinearRegression


if __name__ == '__main__':

    from sklearn.metrics import mean_squared_error

    # load dataframe
    df = pd.read_csv('./are_blue_pills_magics.csv')

    # extract x and y
    x = np.array(df['Micrograms']).reshape(-1,1)
    y = np.array(df['Score']).reshape(-1,1)

    # perform a linear regression
    mlr1 = MyLinearRegression(np.array([[89.0], [-8]]), max_iter=150000)
    mlr2 = MyLinearRegression(np.array([[89.0], [-6]]), max_iter=150000)

    # 1. predict
    y_hat1 = mlr1.predict_(x)
    y_hat2 = mlr2.predict_(x)
    print(f'Prediction with thetas [[89.0], [-8]] :\n{y_hat1}\n')
    print(f'Prediction with thetas [[89.0], [-6]] :\n{y_hat2}\n')

    print(mlr1.mse_(y, y_hat1))
    # 57.60304285714282

    print(mean_squared_error(y, y_hat1))
    # 57.603042857142825

    print(mlr2.mse_(y, y_hat2))
    # 232.16344285714285

    print(mean_squared_error(y, y_hat2))
    # 232.16344285714285

    # 2. evaluate : estimate loss
    loss = mlr1.loss_(y, y_hat1)
    print(f'\nLoss for first prediction: {loss}\n') 

    # 3. improve : fit model
    thetas = mlr1.fit_(x, y)
    print(f'Thetas after fitting :\n{thetas}\n')
    y_hat1 = mlr1.predict_(x)
    print(f'Prediction after fitting :\n{y_hat1}\n')
    loss = mlr1.loss_(y, y_hat1)
    print(f'Loss after fitting : {loss}\n')

    # 3bis. plot loss function
    # plt.figure()
    # plt.title('Loss')
    # loop on different theta value and calculate the loss
    # for theta1 in np.linspace(-8, mlr1.thetas[1][0], 50):
    #     ones_column = np.full((x.shape[0], 1), 1.)
    #     x_matrix = np.hstack((ones_column, x))
    #     # 1. dot product with thetas
    #     y_hat = x_matrix.dot(np.array([mlr1.thetas[0][0], theta1]))
    #     loss = float((((y_hat - y).T.dot(y_hat - y)) / (2 * y.shape[0]))[0][0])
    #     plt.plot(theta1, loss, color='blue', ls='--')
    # plt.xlabel('θ₁')
    # plt.ylabel('Cost function J(θ₀,θ₁)')
    # plt.show()

    # 4. plotting the final result
    plt.figure()
    plt.title('Final result')
    plt.scatter(x, y, marker='o')
    plt.plot(x, y_hat1, color='green', ls='--')
    plt.grid(visible=True)
    plt.xlabel('Quantity of blue pills (in migrograms)')
    plt.ylabel('Space driving score')
    plt.show()


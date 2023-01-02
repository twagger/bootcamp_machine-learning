"""Linear model module"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, '../ex03/')
from my_linear_regression import MyLinearRegression


if __name__ == '__main__':

    from sklearn.metrics import mean_squared_error

    # load dataframe
    df = pd.read_csv('./are_blue_pills_magics.csv')

    # extract x and y
    x = np.array(df['Micrograms']).reshape(-1,1)
    y = np.array(df['Score']).reshape(-1,1)

    # create 2 models
    mlr1 = MyLinearRegression(np.array([[89.0], [-8]]), max_iter=150000)
    mlr2 = MyLinearRegression(np.array([[89.0], [-6]]), max_iter=150000)

    # 1. predict with each model
    y_hat1 = mlr1.predict_(x)
    y_hat2 = mlr2.predict_(x)
    print(f'Prediction with thetas [[89.0], [-8]] :\n{y_hat1}\n')
    print(f'Prediction with thetas [[89.0], [-6]] :\n{y_hat2}\n')

    assert mlr1.mse_(y, y_hat1) == 57.603042857142825

    assert mean_squared_error(y, y_hat1) == 57.603042857142825

    assert mlr2.mse_(y, y_hat2) == 232.16344285714285

    assert mean_squared_error(y, y_hat2) == 232.16344285714285

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

    # 3bis. plot loss function : to finish
    plt.figure()
    plt.title('Loss')
    # loop on different theta value and calculate the loss
    c = 0
    for theta_0 in np.linspace(80, 100, 6):
        J_theta_0 = []
        thetas_1 = []
        for theta_1 in np.linspace(-300, 200, 2500):
            # create a model with the proper thetas
            mlr_temp = MyLinearRegression(np.array([theta_0, theta_1]))
            # get the loss for each value of theta_1 (theta_0 is fixed)
            J_theta_0.append(mlr_temp.loss_(y, mlr_temp.predict_(x)))
            thetas_1.append(theta_1)
        # plot one curve with all theta_1 values and loss for a theta_0
        plt.plot(thetas_1, J_theta_0, c=(c/10 + 0.3, c/10 + 0.3, c/10 + 0.3),
                 label=f'$J(\\theta_0=c_{c},\\theta_1)$')
        # next theta_0 value
        c += 1
    plt.xlim([-15, -3])
    plt.ylim([10, 150])
    plt.grid(visible=True)
    plt.xlabel('$θ_1$')
    plt.ylabel('$Cost function J(θ_0,θ_1)$')
    plt.legend(loc='lower right')
    plt.show()

    # 4. plotting the final result
    plt.figure()
    plt.title('Final result')
    plt.scatter(x, y, marker='o', label='$S_{true}(pills)$')
    plt.scatter(x, y_hat1, marker='x', c='lime', label='$S_{predict}(pills)$')
    plt.plot(x, y_hat1, c='lime', ls='--')
    plt.grid(visible=True)
    plt.xlabel('Quantity of blue pills (in migrograms)')
    plt.ylabel('Space driving score')
    plt.legend()
    plt.show()

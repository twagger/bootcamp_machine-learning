"""Ploting curves module"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# x = np.arange(1,11).reshape(-1,1)
# y = np.array([[ 1.39270298],
#               [ 3.88237651],
#               [ 4.37726357],
#               [ 4.63389049],
#               [ 7.79814439],
#               [ 6.41717461],
#               [ 8.63429886],
#               [ 8.19939795],
#               [10.37567392],
#               [10.68238222]])

# plt.scatter(x,y)
# plt.show()

# # imports
# import sys
# sys.path.insert(1, '../ex06/')
# from polynomial_model import add_polynomial_features
# sys.path.insert(1, '../ex04/')
# from mylinearregression import MyLinearRegression as MyLR

# # Build the model:
# x_ = add_polynomial_features(x, 3)
# my_lr = MyLR(np.ones(4).reshape(-1,1), alpha = 2.5e-6, max_iter = 100000)

# # Fit the model
# my_lr.fit_(x_, y)

# # Plot:
# ## To get a smooth curve, we need a lot of data points
# continuous_x = np.arange(1, 10.01, 0.01).reshape(-1,1)
# x_ = add_polynomial_features(continuous_x, 3)
# y_hat = my_lr.predict_(x_)
# plt.scatter(x, y)
# plt.plot(continuous_x, y_hat, color='orange')
# plt.show()

if __name__ == "__main__":

    # imports
    import sys
    sys.path.insert(1, '../ex06/')
    from polynomial_model import add_polynomial_features
    sys.path.insert(1, '../ex04/')
    from mylinearregression import MyLinearRegression as MyLR

    # Reads and loads are_blue_pills_magics.csv dataset
    data = pd.read_csv("./are_blue_pills_magics.csv")
    X = np.array(data['Micrograms']).reshape((-1, 1))
    Y = np.array(data['Score']).reshape((-1, 1))

    # Trains six separate Linear Regression models with polynomial hypothesis
    # with degrees ranging from 1 to 6
    mse_data = []
    thetas = []
    alphas = []
    max_iters = []
    for i in range(6):
        
        # init params
        theta = np.ones(i + 2).reshape(-1,1)
        alpha = 2.5e-5
        max_iter = 500000
        if i == 3:
            theta = np.array([[-20.0],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
            alpha = 2.5e-8
            max_iter = 500000
        if i == 4:
            theta = np.array([[1140.0],[ -1850],[ 1110],[ -305],[ 40],
                              [ -2]]).reshape(-1, 1)
            alpha = 2.5e-8
            max_iter = 500000
        if i == 5:
            theta = np.array([[9110.0],[ -18015],[ 13400],[ -4935],[ 966],
                              [ -96.4],[ 3.86]]).reshape(-1, 1)
            alpha = 2.5e-10
            max_iter = 1000000
        # add polynomial data
        x_ = add_polynomial_features(X, i + 1)
        # create the model
        my_lr = MyLR(theta, alpha = alpha, max_iter = max_iter)
        # fit the model
        my_lr.fit_(x_, Y)
        # predict
        y_hat = my_lr.predict_(x_)
        # Evaluates and prints evaluation score (MSE) of each of the six models
        mse = my_lr.mse_(Y, y_hat)
        mse_data.append(mse)
        print(mse)
        # saving parameters to reproduce the models for plotting
        thetas.append(my_lr.thetas)
        alphas.append(my_lr.alpha)
        max_iters.append(my_lr.max_iter)


    # Plots a bar plot showing the MSE score of the models in function of the
    # polynomial degree of the hypothesis,
    mse_plot = np.array(mse_data).reshape((-1,1))
    mse_plot = np.hstack((np.arange(1,7).reshape(-1,1), mse_plot))
    plt.figure()
    plt.title('MSE score of the models / polynomial degree of the hypothesis')
    plt.bar(mse_plot[:, 0], mse_plot[:, 1])
    plt.xlabel('Polynomial degree of the hypothesis')
    plt.ylabel('MSE score')
    plt.show()

    # Plots the 6 models and the data points on the same figure.
    # Use lineplot style for the models and scaterplot for the data points.
    # Add more prediction points to have smooth curves for the models.
    plt.figure()
    plt.title('All hypothesis against training dataset')
    plt.scatter(X, Y)
    plt.xlabel('Micrograms')
    plt.ylabel('Score')
    # loop for each hypothesis
    for i in range(6):
        continuous_x = np.arange(1, 10.01, 0.01).reshape(-1,1)
        x_ = add_polynomial_features(continuous_x, i + 1)
        my_lr =  MyLR(thetas[i], alpha = alphas[i], max_iter = max_iters[i])
        y_hat = my_lr.predict_(x_)
        plt.plot(continuous_x, y_hat, label=f'{i + 1} polynomial degree(s)')
        plt.ylim([0, 80])
        plt.legend()
    plt.show()


"""
Space avocado module
This program performs the training of all the models and save the parameters
of the different models into a file.
In models.csv are the parameters of all the models I have explored and trained.
"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import sys
import os
import inspect
import itertools
# nd arrays and dataframes + csv import
import csv
import numpy as np
import pandas as pd
# plot
import matplotlib.pyplot as plt
# wrap / decorators
from functools import wraps
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex06'))
from ridge import MyRidge
from benchmark_train import ModelWithInfos, data_spliter, \
                            add_polynomial_features, polynomial_matrix, \
                            combined_features, normalize_xset, z_score


# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
# Global params
max_iter = 1000
alpha = 1e-1
cols = ['w', 'w2', 'w3', 'w4', 'p', 'p2', 'p3', 'p4', 't', 't2', 't3', 't4',
        'wp', 'wp2', 'wp3', 'wp4', 'wt', 'wt2', 'wt3', 'wt4', 'pt', 'pt2',
        'pt3', 'pt4', 'wpt', 'wpt2', 'wpt3', 'wpt4']

# main program
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 1. Data import
    # -------------------------------------------------------------------------
    # read the dataset and do basic tests on it in case of error
    try:
        df = pd.read_csv("./space_avocado.csv")
    except:
        print("Error when trying to read dataset", file=sys.stderr)
        sys.exit()

    # check that the expected columns are here and check their type
    if not set(['weight', 'prod_distance',
                'time_delivery', 'target']).issubset(df.columns):
        print("Missing columns", file=sys.stderr)
        sys.exit()
    if (df.weight.dtype != float or df.prod_distance.dtype != float
            or df.time_delivery.dtype != float or df.target.dtype != float):
        print("Wrong column type", file=sys.stderr)
        sys.exit()

    # set X and y
    X = np.array(df[['weight', 'prod_distance',
                     'time_delivery']]).reshape((-1, 3))
    y = np.array(df['target']).reshape((-1, 1))

    # -------------------------------------------------------------------------
    # 2. Data preparation : augmentation (no need to clean here)
    # -------------------------------------------------------------------------
    # combine data
    x_comb = combined_features(X, max=3)
    # add polynomial features up to degree 4
    x_poly = polynomial_matrix(x_comb, 4)
    # normalize data to ease thetas optimization through gradient descent
    x_norm = normalize_xset(x_poly)
    # switch back to dataframe and relabel columns to ease feature selection
    # during model training
    df = pd.DataFrame(data=x_norm, columns=cols)

    # -------------------------------------------------------------------------
    # 3. Create model list and load models from models.csv in it
    # -------------------------------------------------------------------------
    models = []

    with open('models.csv', 'r') as file:
        reader = csv.DictReader(file) # DictRader will skip the header row
        num = 0
        for i, row in enumerate(reader):
            if i % 6 == 0:
                num += 1
            thetas = np.array([float(theta) for theta 
                               in row['thetas'].split(',')]).reshape(-1, 1)
            features = list([str(feat) for feat in row['features'].split(',')])
            model = MyRidge(thetas, alpha=float(row['alpha']),
                            max_iter=int(row['max_iter']),
                            lambda_=float(row['lambda']))
            models.append(ModelWithInfos(num=num, m=model,
                                         features=features,
                                         loss=float(row['loss']),
                                         train_loss=float(row['train_loss'])))

    # -------------------------------------------------------------------------
    # 4. Train the best model (the last of the file)
    # -------------------------------------------------------------------------
    # pick the best model from the file
    best_model = models[-1]
    # update the model hyperparameters for a new training from scratch
    best_model.m.thetas = np.random.rand(len(best_model.m.thetas), 1)
    best_model.m.alpha = alpha
    best_model.m.max_iter = max_iter
    best_model.loss = None
    best_model.train_loss = None
    best_model.m.lambda_ = 0 # no regularization

    # train the model with X_train and test with X_test (no cross validation
    # here, the best model has already been selected)
    features = best_model.features
    X = np.array(df[features]).reshape((-1, len(features)))
    X_train, X_test, y_train, y_test = data_spliter(X, y, 0.8)
    # train with different lambdas and store the y_hat and loss per lambda
    # variation for the same model
    y_hat = []
    loss = []
    for lambda_ in np.arange(0.0, 1.2, 0.2):
        best_model.m.set_params_(lambda_ = lambda_)
        best_model.m.fit_(X_train, y_train)
        # predict on test set
        y_hat_test = best_model.m.predict_(X_test)
        # calculate loss
        loss.append(best_model.m.loss_(y_test, y_hat_test))
        # store the predictions
        y_hat.append(y_hat_test)

    # -------------------------------------------------------------------------
    # 5 : Sort the models
    # -------------------------------------------------------------------------
    models = sorted(models[:-1], key=lambda x: x.num)

    # -------------------------------------------------------------------------
    # 6. Plot comparison diagrams and prediction of best model
    # -------------------------------------------------------------------------
    # plot the loss variation for every model with respect to lambda value
    plt.figure()
    plt.title('Loss per model depending on regularization lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    for i in range(0, len(models), 6): # loop on every model, step 5
        model_x = np.empty((0, 1))
        model_y = np.empty((0, 1))
        label = f'{((i / 6) % 4) + 1} degree(s)' \
                f'{", with combined params" if models[i].num < 5 else ""}'
        for j in range(6):
            model_x = np.append(model_x, models[i + j].m.lambda_)
            model_y = np.append(model_y, models[i + j].loss)
        plt.plot(model_x, model_y, label=label)
    plt.legend()
    plt.show()

    # plot 3 scatter plots : prediction vs true price with the best model and
    # different lambdas
    fig, axs = plt.subplots(3, 6)
    fig.suptitle('Prediction versus true price')
    for lambda_ in range(6):

        axs[0, lambda_].set_title(f'Lambda = {lambda_ * 0.2:.1f}')

        axs[0, lambda_].set_xlabel = 'Weight'
        axs[0, lambda_].scatter(X_test[:, features.index('w')], y_test,
                                label='True')
        axs[0, lambda_].scatter(X_test[:, features.index('w')], y_hat[lambda_],
                                label='Predicted', marker='x')

        axs[1, lambda_].set_xlabel('Production distance')
        axs[1, lambda_].scatter(X_test[:, features.index('p')], y_test,
                                label='True')
        axs[1, lambda_].scatter(X_test[:, features.index('p')], y_hat[lambda_],
                                label='Predicted', marker='x')

        axs[2, lambda_].set_xlabel('Time delivery')
        axs[2, lambda_].scatter(X_test[:, features.index('t')], y_test,
                                label='True')
        axs[2, lambda_].scatter(X_test[:, features.index('t')], y_hat[lambda_],
                                label='Predicted', marker='x')

    # y label
    for ax in axs.flat:
        ax.set(ylabel='Price')

    # draw
    plt.legend()
    plt.show()

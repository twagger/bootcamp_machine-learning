"""
Benchmark train module
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
# multi-threading
import concurrent.futures
# nd arrays and dataframes + csv import
import csv
import numpy as np
import pandas as pd
# plot
import matplotlib.pyplot as plt
# progress bar
from tqdm import tqdm
# wrap / decorators
from functools import wraps
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex06'))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex08'))
from my_logistic_regression import MyLogisticRegression as MyLogR
from benchmark_train import ModelWithInfos, data_spliter, \
                            add_polynomial_features, polynomial_matrix, \
                            normalize_xset, z_score, tf_metrics, \
                            precision_score_, recall_score_, f1_score_


# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
# Global params
max_iter = 10000
alpha = 1e-2
cols = ['w', 'w2', 'w3', 'h', 'h2', 'h3', 'd', 'd2', 'd3']


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 1. Data loading
    # -------------------------------------------------------------------------
    # load data from csv
    try:
        df_features = pd.read_csv("./solar_system_census.csv")
        df_labels = pd.read_csv("./solar_system_census_planets.csv")
    except:
        print("error when trying to read datasets", file=sys.stderr)
        sys.exit()

    # concatenate the data into one dataframe
    df = pd.concat([df_features[['weight', 'height', 'bone_density']],
                    df_labels['Origin']], axis=1)

    # check columns types
    if (df.weight.dtype != float or df.height.dtype != float
            or df.bone_density.dtype != float or df.Origin.dtype != float):
        print("wrong column type in dataset", file=sys.stderr)
        sys.exit()

    # set X and y
    X = np.array(df[['weight', 'height', 'bone_density']]).reshape((-1, 3))
    y = np.array(df['Origin']).reshape((-1, 1))

    # -------------------------------------------------------------------------
    # 2. Data preparation (no cleaning, the dataset is already ok)
    # -------------------------------------------------------------------------
    # normalize to ease gradient descent
    X = normalize_xset(X)

    # add polynomial features up to degree 3 : w, w2, w3, h, h2, h3, d, d2, d3
    X_poly = polynomial_matrix(X, 3)

    # switch back to dataframe and relabel columns to ease feature selection
    # during model training
    df = pd.DataFrame(data=X_poly, columns=cols)

    # -------------------------------------------------------------------------
    # 3. Split data : training set / test set (80/20)
    # -------------------------------------------------------------------------
    # split the dataset into a training set and a test set > 80 / 20
    X_train, X_test, y_train, y_test = data_spliter(X_poly, y, 0.8)

    # -------------------------------------------------------------------------
    # 4. Create model list and load models from models.csv in it
    # -------------------------------------------------------------------------
    models = []

    with open('models.csv', 'r') as file:
        reader = csv.DictReader(file) # DictRader will skip the header row
        num = -1
        deg = 1e-2
        for i, row in enumerate(reader):
            if i % 4 == 0:
                num += 1
            if i % 24 == 0:
                deg *= 100
            thetas = np.array([float(theta) for theta 
                               in row['thetas'].split(',')]).reshape(-1, 1)
            features = list([str(feat) for feat in row['features'].split(',')])
            model = MyLogR(thetas, alpha=float(row['alpha']),
                            max_iter=int(row['max_iter']),
                            lambda_=float(row['lambda']))
            models.append(ModelWithInfos(num=num + deg, m=model,
                                         features=features,
                                         f1_score=float(row['f1_score'])))

    # -------------------------------------------------------------------------
    # 4. Train the best model (the last of the file)
    # -------------------------------------------------------------------------
    # split data (train and test) and train one multiclass model (check previous module)

    # -------------------------------------------------------------------------
    # 9. Evaluate: fraction of correct predictions over the total number of
    #              predictions based on the test set.
    # -------------------------------------------------------------------------
    # here we cannot use the loss function to evaluate the global loss as it
    # expect prediction between 0 and 1

    # Select only the subset of features associated with the best model
    features = the_model.features
    f_indices = [i for i, feat in enumerate(cols) if feat in features]
    X_test = X_test[:, f_indices]

    # prediction with the best group of models on test set
    predict = np.empty((y_test.shape[0], 0))
    for mod in the_models:
        predict = np.c_[predict, mod.m.predict_(X_test)]
    predict = np.argmax(predict, axis=1).reshape((-1, 1))

    # fraction of correct predictions over the total number of predictions
    # based on the test set.
    nb_pred = len(predict)
    correct_pred = np.sum(y_test == predict)
    print(f'{"fraction of correct prediction ":33}: '
          f'{correct_pred / nb_pred}'
          f' ({correct_pred}/{nb_pred})')

    # -------------------------------------------------------------------------
    # 10. plot f1 score per lambda
    # -------------------------------------------------------------------------
    # sort the models before ploting to regroup then
    models = sorted(models, key=lambda x: x.num)
    # plot the loss curve for every model with respect to lambda
    plt.figure()
    plt.title('F1 per model depending on regularization lambda')
    plt.xlabel('Lambda')
    plt.ylabel('F1 score')
    for i in range(0, len(models), 24): # loop on every model, step 6 * 4
        model_x = np.empty((0, 1))
        model_y = np.empty((0, 1))
        label = f'{((i / 24) % 3) + 1} degree(s)'
        for j in range(24):
            model_x = np.append(model_x, models[i + j].m.lambda_)
            model_y = np.append(model_y, models[i + j].f1_score)
        plt.plot(model_x, model_y, label=label)
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------
    # 11. plot 3 scatter plots : prediction vs true citizenship per feature
    # -------------------------------------------------------------------------
    # plot
    plt.figure()
    plt.title('Prediction versus true citizenship')
    plt.ylabel('citizenship')
    plt.subplot(1,3,1)
    plt.xlabel('Weight')
    plt.scatter(X_test[:, features.index('w')], y_test, label='True')
    plt.scatter(X_test[:, features.index('w')], predict, label='Predicted',
                marker='x')
    plt.subplot(1,3,2)
    plt.xlabel('Height')
    plt.scatter(X_test[:, features.index('h')], y_test, label='True')
    plt.scatter(X_test[:, features.index('h')], predict, label='Predicted',
                marker='x')
    plt.subplot(1,3,3)
    plt.xlabel('Bone density')
    plt.scatter(X_test[:, features.index('d')], y_test, label='True')
    plt.scatter(X_test[:, features.index('d')], predict, label='Predicted',
                marker='x')
    plt.legend()
    plt.show()

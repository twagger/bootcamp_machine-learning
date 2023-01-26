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
                            precision_score_, recall_score_, f1_score_, \
                            train_model, accuracy_score_


# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
# Global params
max_iter = 100000
alpha = 1e-1
cols = ['w', 'w2', 'w3', 'h', 'h2', 'h3', 'd', 'd2', 'd3']
planets = ['Venus', 'Earth', 'Mars', 'Belt']
colors = ['hotpink', 'deepskyblue', 'orangered', 'gold']


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
    # 5. Train the best model (the last of the file)
    # -------------------------------------------------------------------------
    # extract the best model (group of 4)
    the_models = models[-4:]

    # create train and test label vectors with only 2 possibilities to train
    # the sub-models
    y_trains = []
    y_tests = []
    for zipcode in range(4):
        relabel_log = np.vectorize(lambda x: 1 if x == zipcode else 0)
        y_trains.append(relabel_log(y_train))
        y_tests.append(relabel_log(y_test))

    # train models
    for i in range(len(the_models)):
        # pick random thetas to re-train the model and reset hyperparameters
        the_models[i].m.thetas = np.random.rand(the_models[i].m.thetas.shape[0], 1)
        the_models[i].m.alpha = alpha
        the_models[i].m.max_iter = max_iter
        the_models[i].m.f1_score = 0
        # train model
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(train_model, the_models[i], X_train, y_trains[i])

    # -------------------------------------------------------------------------
    # 6. Evaluate: evaluate models with F1-score on the test set
    # -------------------------------------------------------------------------
    # adapt features of X_test according to the best model function
    features = the_models[0].features
    f_indices = [i for i, feat in enumerate(cols) if feat in features]
    X_test = X_test[:, f_indices]

    # build a prediction vector the group of 4 models
    predict = np.empty((y_test.shape[0], 0))
    for i in range(4):
        predict = np.c_[predict, the_models[i].m.predict_(X_test)]
    predict = np.argmax(predict, axis=1).reshape((-1, 1))
    # compute the f1-score for the group
    multiclass_f1_score: float = 0.0
    for label in range(4):
        multiclass_f1_score += f1_score_(predict, y_test, pos_label = label)
    # put the global score on every sub-model
    for i in range(4):
        the_models[i].f1_score = multiclass_f1_score

    # -------------------------------------------------------------------------
    # 7. Evaluate: fraction of correct predictions over the total number of
    #              predictions based on the test set.
    # -------------------------------------------------------------------------
    # here we cannot use the loss function to evaluate the global loss as it
    # expect prediction between 0 and 1

    # prediction with the best group of models on test set
    predict = np.empty((y_test.shape[0], 0))
    for mod in the_models:
        predict = np.c_[predict, mod.m.predict_(X_test)]
    predict = np.argmax(predict, axis=1).reshape((-1, 1))

    # fraction of correct predictions over the total number of predictions
    # based on the test set.
    nb_pred = len(predict)
    correct_pred = np.sum(y_test == predict)
    print(f'{"Correct predictions ":20}: '
          f'{correct_pred / nb_pred}'
          f' ({correct_pred}/{nb_pred})')
    
    # model metrics (accuracy, precision, recall, f1-score)
    print(f"{'Accuracy score : ':20}{accuracy_score_(y_test, predict)}")
    for i, planet in enumerate(planets):
        print(f'{planet.upper():-<40}')
        print(f"{'Precision score : ':20}"
              f"{precision_score_(y_test, predict, pos_label=i)}")
        print(f"{'Recall score : ':20}"
              f"{recall_score_(y_test, predict, pos_label=i)}")
        print(f"{'F1 score : ':20}{f1_score_(y_test, predict, pos_label=i)}")


    # -------------------------------------------------------------------------
    # 8. plot f1 score per lambda
    # -------------------------------------------------------------------------
    # sort the models before ploting to regroup then and delete the 9999 ones
    models = sorted(models[:len(models) - 4], key=lambda x: x.num)
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
    # 9. plot 3 scatter plots : prediction vs true citizenship per feature
    # -------------------------------------------------------------------------
    # plot
    plt.figure()
    plt.title('Prediction versus true citizenship')
    plt.ylabel('citizenship')

    # colors per planet
    true_set = []
    predict_set = []
    for i, planet in enumerate(planets):
        t_set = np.c_[X_test, y_test]
        true_set.append(t_set[t_set[:, -1] == i])
        p_set = np.c_[X_test, predict]
        predict_set.append(p_set[p_set[:, -1] == i])
    
    # Weight
    plt.subplot(1,3,1)
    plt.xlabel('Weight')
    for i, planet in enumerate(planets):
        plt.scatter(true_set[i][:, features.index('w')], true_set[i][:, -1],
                    label=planet, color=colors[i])
        plt.scatter(predict_set[i][:, features.index('w')],
                    predict_set[i][:, -1], label='', marker='x',
                    color='black')
    
    # Height
    plt.subplot(1,3,2)
    plt.xlabel('Height')
    for i, planet in enumerate(planets):
        plt.scatter(true_set[i][:, features.index('h')], true_set[i][:, -1],
                    label=planet, color=colors[i])
        plt.scatter(predict_set[i][:, features.index('h')],
                    predict_set[i][:, -1], label='', marker='x',
                    color='black')

    # Bone density
    plt.subplot(1,3,3)
    plt.xlabel('Bone density')
    for i, planet in enumerate(planets):
        plt.scatter(true_set[i][:, features.index('d')], true_set[i][:, -1],
                    label=planet, color=colors[i])
        plt.scatter(predict_set[i][:, features.index('d')],
                    predict_set[i][:, -1],
                    label=f"{'Predicted' if i == 3 else ''}", marker='x',
                    color='black')
    plt.legend()
    plt.show()

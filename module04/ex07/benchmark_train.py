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
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex01'))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex06'))
from ridge import MyRidge, type_validator, shape_validator


# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
# Global params
max_iter = 10000
alpha = 1e-1
cols = ['w', 'w2', 'w3', 'w4', 'p', 'p2', 'p3', 'p4', 't', 't2', 't3', 't4',
        'wp', 'wp2', 'wp3', 'wp4', 'wt', 'wt2', 'wt3', 'wt4', 'pt', 'pt2',
        'pt3', 'pt4', 'wpt', 'wpt2', 'wpt3', 'wpt4']


# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------
class ModelWithInfos:
    """
    Generic very small class to store model with basic infos such as the name
    of the features, the model itself and maybe more if needed
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# -----------------------------------------------------------------------------
# Helper functions (Normaly I would import them from modules but if is not
#                   allowed in this exercice to add files in this folder)
# -----------------------------------------------------------------------------
# Data splitter
@type_validator
@shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
def data_spliter(x: np.ndarray, y: np.ndarray, proportion: float) -> tuple:
    """
    Shuffles and splits the dataset (given by x and y) into a training and a
    test set, while respecting the given proportion of examples to be kept in
    the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will
            be assigned to the training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        m, n = x.shape
        # join x and y and shuffle
        full_set = np.hstack((x, y))
        np.random.shuffle(full_set)
        # slice the train and test sets
        train_set_len = int(proportion * m)
        x_train = full_set[:train_set_len, :-1].reshape((-1, n))
        x_test = full_set[train_set_len:, :-1].reshape((-1, n))
        y_train = full_set[:train_set_len, -1].reshape((-1, 1))
        y_test = full_set[train_set_len:, -1].reshape((-1, 1))
        return (x_train, x_test, y_train, y_test)
    except:
        return None


# Polynomial features < Copied because the exercice limit the files to use
@type_validator
@shape_validator({'x': ('m', 1)})
def add_polynomial_features(x: np.ndarray, power: int) -> np.ndarray:
    """
    Add polynomial features to vector x by raising its values up to the power
    given in argument.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        power: has to be an int, the power up to which the components of vector
            x are going to be raised.
    Return:
        The matrix of polynomial features as a numpy.array, of dimension m * n,
            containing the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        result = x.copy()
        for i in range(power - 1):
            result = np.c_[result, x ** (2 + i)]
        return result
    except:
        return None


# Data preparation functions
@type_validator
@shape_validator({'x': ('m', 'n')})
def polynomial_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """return the polynomial matrix for x"""
    try:
        m, n = x.shape
        x_ = np.empty((m, 0))
        for feature in range(n):
            pol = add_polynomial_features(x[:, feature].reshape(-1, 1), degree)
            x_ = np.hstack((x_, pol))
        return x_
    except:
        return None


@type_validator
@shape_validator({'x': ('m', 'n')})
def combined_features(x: np.ndarray, max: int = 2) -> np.ndarray:
    """
    return the combined features matrix for x where every feature is combined
    with each other feature to the maximum level of combination
    """
    try:
        combined = np.copy(x)
        for ii in range(2, max + 1):
            # itertools to generate all unique combination (tuple) of columns
            # I transpose x because itertools.combinations operates on rows
            for subset in itertools.combinations(x.T, ii):
                combined = np.c_[combined, np.prod(subset, axis=0)]
        return combined
    except:
        return None


@type_validator
@shape_validator({'x': ('m', 'n')})
def normalize_xset(x: np.ndarray) -> np.ndarray:
    """Normalize each feature an entire set of data"""
    try:
        m, n = x.shape
        x_norm = np.empty((m, 0))
        for feature in range(n):
            x_norm = np.c_[x_norm, z_score(x[:, feature].reshape(-1, 1))]
        return x_norm
    except:
        return None


# Saving to file functions
@type_validator
@shape_validator({'thetas': ('n', 1)})
def save_training(writer, nb_model: int, form: str, thetas: np.ndarray,
                  alpha: float, max_iter: int, lambda_: float, loss: float,
                  train_loss: float):
    """save the training in csv file"""
    try:
        thetas_str = ','.join([f'{theta[0]}' for theta in thetas])
        writer.writerow([nb_model, form, thetas_str, alpha, max_iter, lambda_,
                         loss, train_loss])
    except:
        return None


# normalization function
@type_validator
@shape_validator({'x': ('m', 1)})
def z_score(x: np.ndarray) -> np.ndarray:
    """
    Computes the normalized version of a non-empty numpy.ndarray using the
        z-score standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x' as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn't raise any Exception.
    """
    try:
        z_score_formula = lambda x, std, mean: (x - mean) / std
        zscore_normalize = np.vectorize(z_score_formula)
        x_prime = zscore_normalize(x, np.std(x), np.mean(x))
        return x_prime
    except:
        return None


# model training function to push it to multithreading
@type_validator
@shape_validator({'X_train': ('m', 'n'), 'Y_train': ('m', 1)})
@shape_validator({'X_val': ('m', 'n'), 'Y_val': ('m', 1)})
def train_model(model: ModelWithInfos, X_train: np.ndarray,
                y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Train the model and save information about its performance"""
    try:
        # Select only the subset of features associated with the model
        features = model.features
        f_indices = [i for i, feat in enumerate(cols) if feat in features]
        X_train = X_train.copy()[:, f_indices]
        X_val = X_val.copy()[:, f_indices]
        # Train model
        model.m.fit_(X_train, y_train)
        # Model metrics : train vs cross validation
        model.train_loss = model.m.loss_(y_train, model.m.predict_(X_train))
        model.loss = model.m.loss_(y_val, model.m.predict_(X_val))
    except:
        return None


# misc tools
@type_validator
def save_model(num: int, models: list, model: MyRidge, features: list):
    """save a model into a dictionnary of models"""
    models.append(ModelWithInfos(num=num, m=model, features=features,
                                 loss=None, train_loss=None))


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
    # - basic features : w, p, t
    # - composite features : wp, wt, pt, wpt
    # - all : w, p, t, wp, wt, pt, wpt
    X_comb = combined_features(X, max=3)

    # add polynomial features up to degree 4
    # degree 1 : w, p, t, wp, wt, pt, wpt
    # degree 2 : w, w2, p, p2, t, t2, wp, wp2, wt, wt2, pt, pt2, wpt, wpt2
    # degree 3 : w, w2, w3, p, p2, p3, t, t2, t3, wp, wp2, wp3, wt, wt2, wt3,
    #            pt, pt2, pt3, wpt, wpt2, wpt3
    # degree 4 : w, w2, w3, w4, p, p2, p3, p4, t, t2, t3, t4, wp, wp2, wp3,
    #            wp4, wt, wt2, wt3, wt4, pt, pt2, pt3, pt4, wpt, wpt2, wpt3,
    #            wpt4
    X_poly = polynomial_matrix(X_comb, 4)

    # normalize data to ease thetas optimization through gradient descent
    X_norm = normalize_xset(X_poly)

    # -------------------------------------------------------------------------
    # 3. Split data : training set / cross validation set / test set (60/20/20)
    # -------------------------------------------------------------------------
    # Split dataset : train / cross validation / test (60/20/20)
    # train / validation + test (60/40)
    X_train, X_val_test, y_train, y_val_test = data_spliter(X_norm, y, 0.6)
    # cross validation / test (50/50)
    X_test, X_val, y_test, y_val = data_spliter(X_val_test, y_val_test, 0.5)

    # -------------------------------------------------------------------------
    # 3. Create models : from basic to complex combination of features
    # -------------------------------------------------------------------------
    # Here I use way less models than in module 02 as gradient descent will
    # minimize the parameters with les effect on the loss
    # I compare the model with and without data augmentation (combined data) to
    # check if it is effective in this case or not ( >> it is not effective).
    models = []

    # apply lambda 0 to 1 (with 0.2 step)
    for lambda_ in np.arange(0.0, 1.2, 0.2):

        # WITH combined
        # all params, including combined, 1 degree
        model = MyRidge(np.random.rand(8, 1), alpha=alpha, max_iter=max_iter,
                        lambda_=lambda_)
        save_model(1, models, model, ['w',
                                      'p',
                                      't',
                                      'wp',
                                      'wt',
                                      'pt',
                                      'wpt'])

        # all params, including combined, 2 degrees
        model = MyRidge(np.random.rand(15, 1), alpha=alpha, max_iter=max_iter,
                        lambda_=lambda_)
        save_model(2, models, model, ['w', 'w2',
                                      'p', 'p2',
                                      't', 't2',
                                      'wp', 'wp2',
                                      'wt', 'wt2',
                                      'pt', 'pt2',
                                      'wpt', 'wpt2'])

        # all params, including combined, 3 degrees
        model = MyRidge(np.random.rand(22, 1), alpha=alpha, max_iter=max_iter,
                        lambda_=lambda_)
        save_model(3, models, model, ['w', 'w2', 'w3',
                                      'p', 'p2', 'p3',
                                      't', 't2', 't3',
                                      'wp', 'wp2', 'wp3',
                                      'wt', 'wt2', 'wt3',
                                      'pt', 'pt2', 'pt3',
                                      'wpt', 'wpt2', 'wpt3'])

        # all params, including combined, 4 degrees
        model = MyRidge(np.random.rand(29, 1), alpha=alpha, max_iter=max_iter,
                        lambda_=lambda_)
        save_model(4, models, model, ['w', 'w2', 'w3', 'w4',
                                      'p', 'p2', 'p3', 'p4',
                                      't', 't2', 't3', 't4',
                                      'wp', 'wp2', 'wp3', 'wp4',
                                      'wt', 'wt2', 'wt3', 'wt4',
                                      'pt', 'pt2', 'pt3', 'pt4',
                                      'wpt', 'wpt2', 'wpt3', 'wpt4'])

        # WITHOUT combined
        # all params, 1 degree
        model = MyRidge(np.random.rand(4, 1), alpha=alpha, max_iter=max_iter,
                        lambda_=lambda_)
        save_model(5, models, model, ['w',
                                      'p',
                                      't'])

        # all params, 2 degrees
        model = MyRidge(np.random.rand(7, 1), alpha=alpha, max_iter=max_iter,
                        lambda_=lambda_)
        save_model(6, models, model, ['w', 'w2',
                                      'p', 'p2',
                                      't', 't2'])

        # all params, 3 degrees
        model = MyRidge(np.random.rand(10, 1), alpha=alpha, max_iter=max_iter,
                        lambda_=lambda_)
        save_model(7, models, model, ['w', 'w2', 'w3',
                                      'p', 'p2', 'p3',
                                      't', 't2', 't3'])

        # all params, 4 degrees
        model = MyRidge(np.random.rand(13, 1), alpha=alpha, max_iter=max_iter,
                        lambda_=lambda_)
        save_model(8, models, model, ['w', 'w2', 'w3', 'w4',
                                      'p', 'p2', 'p3', 'p4',
                                      't', 't2', 't3', 't4'])

    # -------------------------------------------------------------------------
    # 3 Bis : Sort the models
    # -------------------------------------------------------------------------
    models = sorted(models, key=lambda x: x.num)

    # -------------------------------------------------------------------------
    # 4. Train models and store the loss
    # -------------------------------------------------------------------------
    # model training and loss calculation with multithreading
    for model in tqdm(models):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(train_model, model, X_train, y_train, X_val, y_val)

    # -------------------------------------------------------------------------
    # 5. Save all models with their hyperparameters and results
    # -------------------------------------------------------------------------
    # open csv file to save params
    with open('models.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "features", "thetas", "alpha", "max_iter",
                         "lambda", "loss", "train_loss"])

        for ii, model in enumerate(models):
            # saving parameters and results in the models.csv file
            feat_str = ",".join([f'{feat}' for feat in model.features])
            save_training(writer, ii, feat_str, model.m.thetas, alpha,
                          max_iter, model.m.lambda_, model.loss,
                          model.train_loss)

    # -------------------------------------------------------------------------
    # 6. Plot all model's loss and pick best model for space avocado
    # -------------------------------------------------------------------------
    # plot the loss curve for every model with respect to lambda
    plt.figure()
    plt.title('Loss per model depending on regularization lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    for i in range(0, len(models), 6): # loop on every model, step 6
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

    # pick the model with the minimum loss
    the_model = min(models, key=lambda x: x.loss)

    # save it as last in the models.csv file to be retrieved by space avocado
    with open('models.csv', 'a') as file:
        writer = csv.writer(file)
        feat_str = ",".join([f'{feat}' for feat in the_model.features])
        save_training(writer, 9999, feat_str, the_model.m.thetas,
                      alpha, max_iter, the_model.m.lambda_, the_model.loss,
                      the_model.train_loss)

    # plot 3 scatter plots : prediction vs true price with the best model
    # Select only the subset of features associated with the best model
    features = the_model.features
    f_indices = [i for i, feat in enumerate(cols) if feat in features]
    X_test = X_test.copy()[:, f_indices]
    # predict y_hat
    y_hat = the_model.m.predict_(X_test)
    plt.figure()
    plt.title('Prediction versus true price')
    plt.ylabel('Price')
    plt.subplot(1,3,1)
    plt.xlabel('Weight')
    plt.scatter(X_test[:, features.index('w')], y_test, label='True')
    plt.scatter(X_test[:, features.index('w')], y_hat, label='Predicted',
                marker='x')
    plt.subplot(1,3,2)
    plt.xlabel('Production distance')
    plt.scatter(X_test[:, features.index('p')], y_test, label='True')
    plt.scatter(X_test[:, features.index('p')], y_hat, label='Predicted',
                marker='x')
    plt.subplot(1,3,3)
    plt.xlabel('Time delivery')
    plt.scatter(X_test[:, features.index('t')], y_test, label='True')
    plt.scatter(X_test[:, features.index('t')], y_hat, label='Predicted',
                marker='x')
    plt.legend()
    plt.show()

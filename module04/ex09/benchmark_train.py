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
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex08'))
from my_logistic_regression import MyLogisticRegression as MyLogR


# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------
# specific class for the program
class ModelWithInfos:
    """
    Generic very small class to store model with basic infos such as the name
    of the features, the model itself and maybe more if needed
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# -----------------------------------------------------------------------------
# Custom decorators (Normaly I would import them from modules but if is not
#                    allowed in this exercice to add files in this folder)
# -----------------------------------------------------------------------------
# generic type validation based on type annotation in function signature
def type_validator(func):
    """
    Decorator that will rely on the types and attributes declaration in the
    function's signature to check the actual types of the parameter against the
    expected types
    """
    # extract information about the function's parameters and return type.
    sig = inspect.signature(func)
    # preserve name and docstring of decorated function
    @wraps(func)
    def wrapper(*args, **kwargs):
        # map the parameter from signature to their corresponding values
        bound_args = sig.bind(*args, **kwargs)
        # check for each name of param if value has the declared type
        for name, value in bound_args.arguments.items():
            if name in sig.parameters:
                param = sig.parameters[name]
                if (param.annotation != param.empty
                        and not isinstance(value, param.annotation)):
                    print(f"function '{func.__name__}' : " \
                          f"expected type '{param.annotation}' for argument " \
                          f"'{name}' but got {type(value)}.")
                    return None
        return func(*args, **kwargs)
    return wrapper


# -----------------------------------------------------------------------------
# Helper functions (Normaly I would import them from modules but if is not
#                   allowed in this exercice to add files in this folder)
# -----------------------------------------------------------------------------
# data splitter
@type_validator
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
        # shape test
        if not (x.ndim == 2 and y.ndim == 2 and y.shape[0] == x.shape[0]
                and y.shape[1] == 1):
            print('x and y shapes must be (m * n) and (m * 1)',
                  file=sys.stderr)
            return None
        m, n = x.shape
        # join x and y before shuffle
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


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 1. Data loading 
    # -------------------------------------------------------------------------
    try:
        df_features = pd.read_csv("./solar_system_census.csv")
        df_labels = pd.read_csv("./solar_system_census_planets.csv")
    except:
        # At least to catch :
        # FileNotFoundError, PermissionError, pandas.errors.ParserError,
        # pandas.errors.DtypeWarning, pandas.errors.EmptyDataError,
        # pandas.errors.ParserWarning, UnicodeDecodeError:
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
    y = np.array(df['target']).reshape((-1, 1))

    # -------------------------------------------------------------------------
    # 2. Data preparation (no cleaning, the dataset is already ok)
    # -------------------------------------------------------------------------
    # add polynomial features up to degree 3 : w, w2, w3, h, h2, h3, d, d2, d3
    X_poly = polynomial_matrix(X, 3)

    # switch back to dataframe and relabel columns to ease feature selection
    # during model training
    cols = ['w', 'w2', 'w3', 'h', 'h2', 'h3', 'd', 'd2', 'd3']
    df = pd.DataFrame(data=X_poly, columns=cols)

    # -------------------------------------------------------------------------
    # 3. Split data : training set / cross validation set / test set (60/20/20)
    # -------------------------------------------------------------------------
    # split the dataset into a training set and a test set > 80 / 20
    X = np.array(df[['w', 'h', 'd']]).reshape((-1, 3))
    y = np.array(df['o']).reshape((-1, 1))
    X_train, X_test, y_train, y_test = data_spliter(X, y, 0.8)
    # split the train dataset into a training set
    # and a cross validation set > 80 / 20
    X_train, X_val, y_train, y_val = data_spliter(X_train, y_train, 0.8)

    # -------------------------------------------------------------------------
    # 3. Create train and cross validation label vectors with only 2 outputs
    # -------------------------------------------------------------------------
    y_trains = []
    y_vals = []
    for zipcode in range(4):
        relabel_log = np.vectorize(lambda x: 1 if x == zipcode else 0)
        y_trains.append(relabel_log(y_train))
        y_vals.append(relabel_log(y_test))

    # -------------------------------------------------------------------------
    # 4. Train different regularized logistic regression models with a
    #    polynomial hypothesis of degree 3.
    #    Train the models with different λ values: from 0 to 1 (0.2 step)
    # -------------------------------------------------------------------------
    # apply lambda 0 to 1 (with 0.2 step)
    for lambda_ in np.arange(0.0, 1.2, 0.2):
        model = MyLogR(np.random.rand(4, 1), alpha: float = 0.001,
                 max_iter: int = 1000, penalty: str = 'l2',
                 lambda_: float = 1.0)
        save_model(1, models, model, ['w',
                                      'p',
                                      't',
                                      'wp',
                                      'wt',
                                      'pt',
                                      'wpt',])

    # -------------------------------------------------------------------------
    # 3. Model training : Train 4 logistic regression classifiers to
    #       discriminate each class from the others
    # -------------------------------------------------------------------------
    # train a logistic model to predict if the citizen is from the zipcode or
    # not
    sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex06'))
    from my_logistic_regression import MyLogisticRegression as MyLR

    # create models
    models = []
    for _ in range(4):
        models.append(MyLR(np.random.rand(4, 1), alpha=1e-2, max_iter=500000))

    # train models
    for i in range(4):
        models[i].fit_(X_train, y_trains[i])

    # -------------------------------------------------------------------------
    # 4. Predict for each example the class according to each classifiers
    #       and select the one with the highest output probability.
    # -------------------------------------------------------------------------
    predict = np.empty((y_test.shape[0], 0))
    for i in range(4):
        predict = np.c_[predict, models[i].predict_(X_test)]
    predict = np.argmax(predict, axis=1).reshape((-1, 1))

    # -------------------------------------------------------------------------
    # 5. Evaluate: fraction of correct predictions over the total number of
    #       predictions based on the test set.
    # -------------------------------------------------------------------------
    # here we cannot use the loss function to evaluate the global loss as it
    # expect prediction between 0 and 1

    # fraction of correct predictions over the total number of predictions
    # based on the test set.
    nb_pred = len(predict)
    correct_pred = np.sum(y_test == predict)
    print(f'{"fraction of correct prediction ":33}: '
          f'{correct_pred / nb_pred}'
          f' ({correct_pred}/{nb_pred})')

    # -------------------------------------------------------------------------
    # 6. plot 3 scatter plots : prediction vs true citizenship per feature
    # -------------------------------------------------------------------------
    # plot
    plt.figure()
    plt.title('Prediction versus true citizenship')
    plt.ylabel('citizenship')
    plt.subplot(1,3,1)
    plt.xlabel('Weight')
    plt.scatter(X_test[:, 0], y_test, label='True')
    plt.scatter(X_test[:, 0], predict, label='Predicted',
                marker='x')
    plt.subplot(1,3,2)
    plt.xlabel('Height')
    plt.scatter(X_test[:, 1], y_test, label='True')
    plt.scatter(X_test[:, 1], predict, label='Predicted',
                marker='x')
    plt.subplot(1,3,3)
    plt.xlabel('Bone density')
    plt.scatter(X_test[:, 2], y_test, label='True')
    plt.scatter(X_test[:, 2], predict, label='Predicted',
                marker='x')
    plt.legend()
    plt.show()

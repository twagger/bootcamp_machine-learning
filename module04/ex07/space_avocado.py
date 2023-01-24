"""Space avocado module
This program performs the training of all the models and save the parameters
of the different models into a file.
In models.csv are the parameters of all the models I have explored and trained.
"""
# general modules
import sys
import os
import itertools
import inspect
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from functools import wraps

# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex06'))
from ridge import MyRidge

# Global params
max_iter = 1000
alpha = 1e-1


# generic type validation based on type annotation in function signature
def type_validator(func):
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
                    print(f"Expected type '{param.annotation}' for argument " \
                          f"'{name}' but got {type(value)}.")
                    return None
        return func(*args, **kwargs)
    return wrapper


# specific data structure
class ModelWithInfos:
    """
    Generic very small class to store model with basic infos such as the name
    of the features, the model itself and maybe more if needed
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Data splitter < Copied because the exercice limit the files to use
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
        m, n = x.shape
        if y.shape[0] != m or y.shape[1] != 1:
            print('Something went wrong', file=sys.stderr)
            return None
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

    except (ValueError, TypeError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None


# Polynomial features < Copied because the exercice limit the files to use
@type_validator
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
        x = x.reshape((-1, 1))
        # calculation
        result = x.copy()
        for i in range(power - 1):
            result = np.c_[result, x ** (2 + i)]
        return result

    except (ValueError, TypeError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None


# Data preparation functions
@type_validator
def polynomial_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """return the polynomial matrix for x"""
    try:
        m, n = x.shape
        x_ = np.empty((m, 0))
        for feature in range(n):
            x_ = np.hstack((x_, add_polynomial_features(x[:, feature], degree)))
        return x_
    except (ValueError, TypeError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None


@type_validator
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
    except (AttributeError, ValueError, TypeError) as exc:
        print(exc, file=sys.stderr)
        return None


@type_validator
def normalize_xset(x: np.ndarray) -> np.ndarray:
    """Normalize each feature an entire set of data"""
    try:
        x_norm = np.empty((x.shape[0], 0))
        for feature in range(x.shape[1]):
            x_norm = np.hstack((x_norm, z_score(x[:, feature])))
        return x_norm
    except (AttributeError, TypeError) as exc:
        print(exc, file=sys.stderr)
        return None


# Saving to file functions
@type_validator
def save_training(writer, nb_model: int, form: str, thetas: np.ndarray,
                  alpha: float, max_iter: int, lambda_: float, loss: float,
                  train_loss: float):
    """save the training in csv file"""
    try:
        thetas_str = ','.join([f'{theta[0]}' for theta in thetas])
        writer.writerow([nb_model, form, thetas_str, alpha, max_iter, loss,
                         train_loss])
    except (AttributeError, TypeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return None


# normalization function
@type_validator
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
        # shape test
        x = x.reshape((-1, 1))
        # normalization
        z_score_formula = lambda x, std, mean: (x - mean) / std
        zscore_normalize = np.vectorize(z_score_formula)
        x_prime = zscore_normalize(x, np.std(x), np.mean(x))
        return x_prime

    except (ValueError, TypeError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None


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
    x_comb = combined_features(X, max=3)

    # add polynomial features up to degree 4
    # degree 1 : w, p, t, wp, wt, pt, wpt
    # degree 2 : w, w2, p, p2, t, t2, wp, wp2, wt, wt2, pt, pt2, wpt, wpt2
    # degree 3 : w, w2, w3, p, p2, p3, t, t2, t3, wp, wp2, wp3, wt, wt2, wt3,
    #            pt, pt2, pt3, wpt, wpt2, wpt3
    # degree 4 : w, w2, w3, w4, p, p2, p3, p4, t, t2, t3, t4, wp, wp2, wp3,
    #            wp4, wt, wt2, wt3, wt4, pt, pt2, pt3, pt4, wpt, wpt2, wpt3,
    #            wpt4
    x_poly = polynomial_matrix(x_comb, 4)

    # normalize data to ease thetas optimization through gradient descent
    x_norm = normalize_xset(x_poly)

    # switch back to dataframe and relabel columns to ease feature selection
    # during model training
    cols = ['w', 'w2', 'w3', 'w4', 'p', 'p2', 'p3', 'p4', 't', 't2', 't3',
            't4', 'wp', 'wp2', 'wp3', 'wp4', 'wt', 'wt2', 'wt3', 'wt4', 'pt',
            'pt2', 'pt3', 'pt4', 'wpt', 'wpt2', 'wpt3', 'wpt4']
    df = pd.DataFrame(data=x_norm, columns=cols)

    # -------------------------------------------------------------------------
    # 3. Create model list and load models from models.csv in it
    # -------------------------------------------------------------------------
    models = []

    with open('models.csv', 'r') as file:
        reader = csv.DictReader(file) # DictRader will skip the header row
        num = 0
        for i, row in enumerate(reader):
            if (i % 6 == 0):
                num += 1
            thetas = np.array([float(theta)
                               for theta in row['thetas'].split(',')])
            features = list([str(feat) for feat in row['features'].split(',')])
            model = MyRidge(thetas = thetas, alpha=float(row['alpha']),
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

    # train the model with X_train
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
        # calculate loss
        loss.append(best_model.m.loss_(y_test, best_model.m.predict_(X_test)))
        # predict on global dataset
        y_hat.append(best_model.m.predict_(df[best_model.features].to_numpy()))

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
        axs[0, lambda_].scatter(np.array(df['w']).reshape((-1, 1)), y,
                                label='True')
        axs[0, lambda_].scatter(np.array(df['w']).reshape((-1, 1)),
                                y_hat[lambda_], label='Predicted', marker='x')

        axs[1, lambda_].set_xlabel('Production distance')
        axs[1, lambda_].scatter(np.array(df['p']).reshape((-1, 1)), y,
                                label='True')
        axs[1, lambda_].scatter(np.array(df['p']).reshape((-1, 1)),
                                y_hat[lambda_], label='Predicted', marker='x')

        axs[2, lambda_].set_xlabel('Time delivery')
        axs[2, lambda_].scatter(np.array(df['t']).reshape((-1, 1)), y,
                                label='True')
        axs[2, lambda_].scatter(np.array(df['t']).reshape((-1, 1)),
                                y_hat[lambda_], label='Predicted', marker='x')

    # y label
    for ax in axs.flat:
        ax.set(ylabel='Price')

    # draw
    plt.legend()
    plt.show()

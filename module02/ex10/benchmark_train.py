"""
Benchmark train module
This program performs the training of all the models and save the parameters
of the different models into a file.
In models.csv are the parameters of all the models I have explored and trained.
"""
# general modules
import os
import itertools
import concurrent.futures
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# user modules
import sys
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex09'))
from data_spliter import data_spliter
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex07'))
from polynomial_model import add_polynomial_features
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex05'))
from mylinearregression import MyLinearRegression as MyLR

# Global params
max_iter = 10000
alpha = 1e-1
#np.set_printoptions(precision=2)

# specific data structure
class ModelWithInfos:
    """
    Generic very small class to store model with basic infos such as the name
    of the features, the model itself and maybe more if needed
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Data preparation functions
def polynomial_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """return the polynomial matrix for x"""
    x_ = np.empty((x.shape[0], 0))
    for feature in range(x.shape[1]):
        x_ = np.hstack((x_, add_polynomial_features(x[:, feature], degree)))
    return x_


def combined_features(x: np.ndarray, max: int = 2) -> np.ndarray:
    """
    return the combined features matrix for x where every feature is combined
    with each other feature to the maximum level of combination
    """
    combined = np.copy(x)
    for ii in range(2, max + 1):
        # itertools to generate all unique combination (tuple) of columns
        # I transpose x because itertools.combinations operates on rows
        for subset in itertools.combinations(x.T, ii):
            combined = np.c_[combined, np.prod(subset, axis=0)]
    return combined


def normalize_xset(x: np.ndarray) -> np.ndarray:
    """Normalize each feature an entire set of data"""
    x_norm = np.empty((x.shape[0], 0))
    for feature in range(x.shape[1]):
        x_norm = np.hstack((x_norm, z_score(x[:, feature])))
    return x_norm


# Saving to file functions
def save_training(writer, nb_model: int, form: str, thetas: np.ndarray,
                  alpha: float, max_iter: int, loss: float):
    """save the training in csv file"""
    thetas_str = ','.join([f'[{theta[0]}]' for theta in thetas])
    writer.writerow([nb_model, form, thetas_str, alpha, max_iter, loss])


# normalization function
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
        # type test
        if not isinstance(x, np.ndarray):
            print("Something went wrong")
            return None
        # shape test
        x = x.reshape((-1, 1))
        # normalization
        z_score_formula = lambda x, std, mean: (x - mean) / std
        zscore_normalize = np.vectorize(z_score_formula)
        x_prime = zscore_normalize(x, np.std(x), np.mean(x))
        return x_prime

    except (ValueError, TypeError, AttributeError) as exc:
        print(exc)
        return None


# model training function to push it to multithreading
def train_model(model: ModelWithInfos, df):
    """Train the model and save information about its performance"""
    # Select the subset of features associated with the model
    features = model.features
    X = np.array(df[features]).reshape((-1, len(features)))
    # split function to produce X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = data_spliter(X, y, 0.8)
    # Train the model on 80% of the data
    model.m.fit_(X_train, y_train)
    # Test the model against 20% of the data and mesure the loss
    model.loss = model.m.loss_(y_test, model.m.predict_(X_test))


# misc tools
def save_model(models: list, model: MyLR, features: list):
    """save a model into a dictionnary of models"""
    models.append(ModelWithInfos(m=model, features=features, loss=None))


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
    # 2. Data preparation : augmentation
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

    # normalize data
    x_norm = normalize_xset(x_poly)

    # switch back to dataframe and relabel columns to ease future use
    cols = ['w', 'w2', 'w3', 'w4', 'p', 'p2', 'p3', 'p4', 't', 't2', 't3',
            't4', 'wp', 'wp2', 'wp3', 'wp4', 'wt', 'wt2', 'wt3', 'wt4', 'pt',
            'pt2', 'pt3', 'pt4', 'wpt', 'wpt2', 'wpt3', 'wpt4']
    df = pd.DataFrame(data=x_norm, columns=cols)

    # -------------------------------------------------------------------------
    # 3. Create models : from basic to complex combination of feat
    # -------------------------------------------------------------------------
    models = []

    # BASIC
    # one parameter
    w_ = MyLR(np.random.rand(2, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, w_, ['w'])

    p_ = MyLR(np.random.rand(2, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, p_, ['p'])

    t_ = MyLR(np.random.rand(2, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, t_, ['t'])

    # two parameters
    w_p = MyLR(np.random.rand(3, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, w_p, ['w', 'p'])

    w_t = MyLR(np.random.rand(3, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, w_t, ['w', 't'])

    t_p = MyLR(np.random.rand(3, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, t_p, ['t', 'p'])

    # three parameters
    w_p_t = MyLR(np.random.rand(4, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, w_p_t, ['w', 'p', 't'])

    # COMBINED PARAMS
    # one parameter
    # lr_wp = MyLR(np.random.rand(2, 1), alpha=alpha, max_iter=max_iter)
    # lr_wt = MyLR(np.random.rand(2, 1), alpha=alpha, max_iter=max_iter)
    # lr_pt = MyLR(np.random.rand(2, 1), alpha=alpha, max_iter=max_iter)
    # lr_wpt = MyLR(np.random.rand(2, 1), alpha=alpha, max_iter=max_iter)
    # # two parameters (1 combined with 1 non combined)
    # lr_wp_t = MyLR(np.random.rand(3, 1), alpha=alpha, max_iter=max_iter)
    # lr_wt_p = MyLR(np.random.rand(3, 1), alpha=alpha, max_iter=max_iter)
    # lr_tp_w = MyLR(np.random.rand(3, 1), alpha=alpha, max_iter=max_iter)

    # POLYNOMIAL
    # 2 degrees

    # 3 degrees

    # 4 degrees

    # -------------------------------------------------------------------------
    # 4. Train models and store the loss
    # -------------------------------------------------------------------------
    # multithreading for model training and loss calculation
    for model in models:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(train_model, model, df)

    # -------------------------------------------------------------------------
    # 5. Save all models with their hyperparameters and results
    # -------------------------------------------------------------------------
    # open csv file to save params
    with open('models.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "features", "thetas", "alpha", "max_iter",
                         "loss"])

        for ii, model in enumerate(models):
            # saving parameters and results in the models.csv file
            save_training(writer, ii, model.features, model.m.thetas, alpha,
                          max_iter, model.loss)

    # -------------------------------------------------------------------------
    # 6. Pick best model for space avocado
    # -------------------------------------------------------------------------
    # plot different graphs to compare models

    # pick the model with the minimum loss

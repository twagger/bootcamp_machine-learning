"""Space avocado module
This program performs the training of all the models and save the parameters
of the different models into a file.
In models.csv are the parameters of all the models I have explored and trained.
"""
# general modules
import sys
import os
import itertools
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex09'))
from data_spliter import data_spliter
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex07'))
from polynomial_model import add_polynomial_features
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex05'))
from mylinearregression import MyLinearRegression as MyLR

# Global params
max_iter = 1000000
alpha = 1e-1

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
    # type test
    if not isinstance(degree, int):
        print("Something went wrong", file=sys.stderr)
        return None
    try:
        m, n = x.shape
        x_ = np.empty((m, 0))
        for feature in range(n):
            x_ = np.hstack((x_, add_polynomial_features(x[:, feature], degree)))
        return x_
    except (ValueError, TypeError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None

def combined_features(x: np.ndarray, max: int = 2) -> np.ndarray:
    """
    return the combined features matrix for x where every feature is combined
    with each other feature to the maximum level of combination
    """
    # type test
    if not isinstance(max, int):
        print("Something went wrong", file=sys.stderr)
        return None
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
def save_training(writer, nb_model: int, form: str, thetas: np.ndarray,
                  alpha: float, max_iter: int, loss: float):
    """save the training in csv file"""
    # type check
    if (not isinstance(nb_model, int) or not isinstance(form, str)
            or not isinstance(thetas, np.ndarray)
            or not isinstance(alpha, float) or not isinstance(max_iter, int)
            or not isinstance(loss, float)):
        print("Something went wrong", file=sys.stderr)
        return None
    try:
        thetas_str = ','.join([f'{theta[0]}' for theta in thetas])
        writer.writerow([nb_model, form, thetas_str, alpha, max_iter, loss])
    except (AttributeError, TypeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return None


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
        for row in reader:
            thetas = np.array([float(theta)
                               for theta in row['thetas'].split(',')])
            features = list([str(feat) for feat in row['features'].split(',')])
            model = MyLR(thetas, alpha=float(row['alpha']),
                         max_iter=int(row['max_iter']))
            models.append(ModelWithInfos(m=model, features=features,
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

    # train the model with X_train
    features = best_model.features
    X = np.array(df[features]).reshape((-1, len(features)))
    X_train, X_test, y_train, y_test = data_spliter(X, y, 0.8)
    best_model.m.fit_(X_train, y_train)

    # -------------------------------------------------------------------------
    # 5. Evaluate loss
    # -------------------------------------------------------------------------
    best_model.loss = best_model.m.loss_(y_test, best_model.m.predict_(X_test))

    # -------------------------------------------------------------------------
    # 6. Plot comparison diagrams and prediction of best model
    # -------------------------------------------------------------------------
    # plot one graph to compare models (mse per model)
    plt.figure()
    plt.title('Loss per model')
    plt.xlabel('Model')
    plt.ylabel('Loss')
    for ii, model in enumerate(models[:-1]):
        # group models so it is understandable
        if ii < 7:
            label = 'Basic'
            color = 'blue'
        elif ii < 14:
            label = 'Combined'
            color = 'green'
        elif ii < 35:
            label = 'Polynomial'
            color = 'red'
        else:
            label = 'All (polynomial w/ all features, incl. combined)'
            color = 'orange'
        plt.scatter(ii, model.loss,
                    label=label if (ii == 0 or ii == 7 or ii == 14
                                    or ii == 35) else None, c=color)
    plt.legend()
    plt.show()

    # plot the differences between train loss and test loss to check if models
    # are overfitting
    plt.figure()
    plt.title('Difference between train and test set on global loss '
              '(positive = the model is better in training / Negative = '
              'the model is better in test')
    plt.xlabel('Model')
    plt.ylabel('Difference between train and test set on loss')
    for ii, model in enumerate(models[:-1]):
        plt.bar(ii, model.loss - model.train_loss)
    plt.show()

    # plot 3 scatter plots : prediction vs true price
    y_hat = best_model.m.predict_(df[best_model.features].to_numpy())
    plt.figure()
    plt.title('Prediction versus true price')
    plt.ylabel('Price')
    plt.subplot(1,3,1)
    plt.xlabel('Weight')
    plt.scatter(np.array(df['w']).reshape((-1, 1)), y, label='True')
    plt.scatter(np.array(df['w']).reshape((-1, 1)), y_hat, label='Predicted',
                marker='x')
    plt.subplot(1,3,2)
    plt.xlabel('Production distance')
    plt.scatter(np.array(df['p']).reshape((-1, 1)), y, label='True')
    plt.scatter(np.array(df['p']).reshape((-1, 1)), y_hat, label='Predicted',
                marker='x')
    plt.subplot(1,3,3)
    plt.xlabel('Time delivery')
    plt.scatter(np.array(df['t']).reshape((-1, 1)), y, label='True')
    plt.scatter(np.array(df['t']).reshape((-1, 1)), y_hat, label='Predicted',
                marker='x')
    plt.legend()
    plt.show()

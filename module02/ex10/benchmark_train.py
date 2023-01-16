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
    thetas_str = ','.join([f'{theta[0]}' for theta in thetas])
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
    feat_str = ",".join([f'{feat}' for feat in features])
    models.append(ModelWithInfos(m=model, features=feat_str, loss=None))


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
    # 3. Create models : from basic to complex combination of features
    # -------------------------------------------------------------------------
    models = []

    # BASIC
    # one parameter
    model = MyLR(np.random.rand(2, 1), alpha=alpha, max_iter=max_iter)
    [save_model(models, model, [feat]) for feat in ['w',
                                                    'p',
                                                    't']]

    # two parameters
    model = MyLR(np.random.rand(3, 1), alpha=alpha, max_iter=max_iter)
    [save_model(models, model, feat) for feat in [['w', 'p'],
                                                  ['w', 't'],
                                                  ['t', 'p']]]

    # three parameters
    model = MyLR(np.random.rand(4, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, model, ['w', 'p', 't'])

    # COMBINED PARAMS
    # one parameter
    model = MyLR(np.random.rand(2, 1), alpha=alpha, max_iter=max_iter)
    [save_model(models, model, [feat]) for feat in ['wp',
                                                    'wt',
                                                    'pt',
                                                    'wpt']]

    # two parameters (1 combined with 1 non combined)
    model = MyLR(np.random.rand(3, 1), alpha=alpha, max_iter=max_iter)
    [save_model(models, model, feat) for feat in [['wp', 't'],
                                                  ['wt', 'p'],
                                                  ['pt', 'w']]]


    # POLYNOMIAL
    # one parameter, 2 degrees
    model = MyLR(np.random.rand(3, 1), alpha=alpha, max_iter=max_iter)
    [save_model(models, model, feat) for feat in [['w', 'w2'],
                                                  ['p', 'p2'],
                                                  ['t', 't2']]]
    # one parameter, 3 degrees
    model = MyLR(np.random.rand(4, 1), alpha=alpha, max_iter=max_iter)
    [save_model(models, model, feat) for feat in [['w', 'w2', 'w3'],
                                                  ['p', 'p2', 'p3'],
                                                  ['t', 't2', 't3']]]

    # one parameter, 4 degrees
    model = MyLR(np.random.rand(5, 1), alpha=alpha, max_iter=max_iter)
    [save_model(models, model, feat) for feat in [['w', 'w2', 'w3', 'w4'],
                                                  ['p', 'p2', 'p3', 'p4'],
                                                  ['t', 't2', 't3', 't4']]]

    # two parameters, 2 degrees
    model = MyLR(np.random.rand(5, 1), alpha=alpha, max_iter=max_iter)
    [save_model(models, model, feat) for feat in [['w', 'w2', 'p', 'p2'],
                                                  ['w', 'w2', 't', 't2'],
                                                  ['t', 't2', 'p', 'p2']]]

    # two parameters, 3 degrees
    model = MyLR(np.random.rand(7, 1), alpha=alpha, max_iter=max_iter)
    [save_model(models, model, feat) for feat in [['w', 'w2', 'w3',
                                                   'p', 'p2', 'p3'],
                                                  ['w', 'w2', 'w3',
                                                   't', 't2', 't3'],
                                                  ['t', 't2', 't3',
                                                   'p', 'p2', 'p3']]]

    # two parameters, 4 degrees
    model = MyLR(np.random.rand(9, 1), alpha=alpha, max_iter=max_iter)
    [save_model(models, model, feat) for feat in [['w', 'w2', 'w3', 'w4',
                                                   'p', 'p2', 'p3', 'p4'],
                                                  ['w', 'w2', 'w3', 'w4',
                                                   't', 't2', 't3', 't4'],
                                                  ['t', 't2', 't3', 't4',
                                                   'p', 'p2', 'p3', 'p4']]]
    # three parameters, 2 degrees
    model = MyLR(np.random.rand(7, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, model, ['w', 'w2',
                               'p', 'p2',
                               't', 't2'])

    # three parameters, 3 degrees
    model = MyLR(np.random.rand(10, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, model, ['w', 'w2', 'w3',
                               'p', 'p2', 'p3',
                               't', 't2', 't3'])

    # three parameters, 4 degrees
    model = MyLR(np.random.rand(13, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, model, ['w', 'w2', 'w3', 'w4',
                               'p', 'p2', 'p3', 'p4',
                               't', 't2', 't3', 't4'])

    # all params, including combined, 1 degree
    model = MyLR(np.random.rand(8, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, model, ['w',
                               'p',
                               't',
                               'wp',
                               'wt',
                               'pt',
                               'wpt',])

    # all params, including combined, 2 degrees
    model = MyLR(np.random.rand(15, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, model, ['w', 'w2',
                               'p', 'p2',
                               't', 't2',
                               'wp', 'wp2',
                               'wt', 'wt2',
                               'pt', 'pt2',
                               'wpt', 'wpt2',])

    # all params, including combined, 3 degrees
    model = MyLR(np.random.rand(22, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, model, ['w', 'w2', 'w3',
                               'p', 'p2', 'p3',
                               't', 't2', 't3',
                               'wp', 'wp2', 'wp3',
                               'wt', 'wt2', 'wt3',
                               'pt', 'pt2', 'pt3',
                               'wpt', 'wpt2', 'wpt3',])

    # all params, including combined, 4 degrees
    model = MyLR(np.random.rand(29, 1), alpha=alpha, max_iter=max_iter)
    save_model(models, model, ['w', 'w2', 'w3', 'w4',
                               'p', 'p2', 'p3', 'p4',
                               't', 't2', 't3', 't4',
                               'wp', 'wp2', 'wp3', 'wp4',
                               'wt', 'wt2', 'wt3', 'wt4',
                               'pt', 'pt2', 'pt3', 'pt4',
                               'wpt', 'wpt2', 'wpt3', 'wpt4'])

    # -------------------------------------------------------------------------
    # 4. Train models and store the loss
    # -------------------------------------------------------------------------
    # model training and loss calculation with multithreading
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
    # 6. Plot all model's loss and pick best model for space avocado
    # -------------------------------------------------------------------------
    # plot one graph to compare models (mse per model)
    plt.figure()
    plt.title('Loss per model')
    plt.xlabel('Model')
    plt.ylabel('Loss')
    for ii, model in enumerate(models):
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
            label = 'All (polynomial with all features, including combined)'
            color = 'orange'
        plt.scatter(ii, model.loss,
                    label=label if (ii == 0 or ii == 7 or ii == 14
                                    or ii == 35) else None, c=color)
    plt.legend()
    plt.show()

    # pick the model with the minimum loss
    the_model = min(models, key=lambda x: x.loss)

    # save it as last in the models.csv file to be retrieved by space avocado
    with open('models.csv', 'w') as file:
        writer = csv.writer(file)
        save_training(writer, ii, the_model.features, the_model.m.thetas,
                      alpha, max_iter, the_model.loss)

    # plot 3 scatter plots : prediction vs true price
    y_hat = the_model.m.predict_(df[the_model.features].to_numpy())
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

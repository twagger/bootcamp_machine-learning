"""
Benchmark train module
This program performs the training of all the models and save the parameters
of the different models into a file.
In models.csv are the parameters of all the models I have explored and trained.
"""
# general modules
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
# user modules
import sys
sys.path.insert(1, '../ex09/')
from data_spliter import data_spliter
sys.path.insert(1, '../ex07/')
from polynomial_model import add_polynomial_features
sys.path.insert(1, '../ex05')
from mylinearregression import MyLinearRegression as MyLR



# Helper functions
def polynomial_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """return the polynomial matrix for x"""
    x_ = np.empty((x.shape[0], 0))
    for feature in range(x.shape[1]):
        x_ = np.hstack((x_, add_polynomial_features(x[:, feature], degree)))
    return x_


def save_training(writer, nb_model: int, degree: int, thetas: np.ndarray,
                  alpha: float, max_iter: int, loss: float):
    """save the training in csv file"""
    f_thetas = '[' + ', '.join(str(t) for t in my_lr.thetas.flatten()) + ']'
    writer.writerow([nb_model, degree, f_thetas, alpha, max_iter, loss])


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


def minmax(x: np.ndarray) -> np.ndarray:
    """
    Computes the normalized version of a non-empty numpy.ndarray using the
        min-max standardization.
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
        min_max_formula = lambda x, min, max: (x - min) / (max - min)
        minmax_normalize = np.vectorize(min_max_formula)
        x_prime = minmax_normalize(x, np.min(x), np.max(x))
        return x_prime

    except (ValueError, TypeError, AttributeError) as exc:
        print(exc)
        return None


def normalize_xset(x: np.ndarray) -> np.ndarray:
    """Normalize each feature an entire set of data"""
    x_norm = np.empty((x.shape[0], 0))
    for feature in range(x.shape[1]):
        x_norm = np.hstack((x_norm, minmax(x[:, feature])))
    return x_norm


def plot_features(x: np.ndarray, y: np.ndarray):
    """plot each x feature versus y"""
    plt.figure()
    for feature in range(x.shape[1]):
        plt.subplot(1, x.shape[1], feature + 1)
        plt.grid()
        plt.title(f'{dataset.columns.tolist()[feature + 1]}')
        plt.scatter(x[:, feature], y)
        plt.ylabel(f'{dataset.columns.tolist()[-1]}')
    plt.show()


# main program
if __name__ == "__main__":

    # read the dataset
    dataset = pd.read_csv("./space_avocado.csv")

    # open csv file to save params
    with open('models.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "degree", "thetas", "alpha", "max_iter",
                         "split"])

        # create x matrix and y vector
        x = np.array(dataset[['weight', 'prod_distance',
                            'time_delivery']]).reshape((-1, 3))
        y = np.array(dataset['target']).reshape((-1, 1))
        x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

        # feature scaling to help gradient descend
        x_train_norm = normalize_xset(x_train)

        # plot every feature with Y to have a global understanding
        plot_features(x_train, y_train)

        # Trains 4 separate Linear Regression models with polynomial
        # hypothesis with degrees ranging from 1 to 4
        nb_model = 0
        for degree in range(1, 3):

            # global params
            theta = np.zeros((degree * x_train.shape[1] + 1, 1))
            alpha = 0.001
            max_iter = 1
            # specific params
            if degree == 1:
                alpha = 0.666
                max_iter = 1

            # add polynomial features for each feature
            x_train_ = polynomial_matrix(x_train_norm, degree)

            # create the model
            my_lr = MyLR(theta, alpha = alpha, max_iter = max_iter)
            nb_model += 1

            # fit the model and store loss each time to observe progress
            losses = []
            it = []
            for iteration in range(50000):
                my_lr.fit_(x_train_, y_train)
                x_test_ = polynomial_matrix(x_test, degree)
                y_hat_test = my_lr.predict_(x_test_)
                losses.append(my_lr.loss_(y_test, y_hat_test))
                it.append(iteration)

            # plot the learning curve
            plt.figure()
            plt.grid()
            plt.title(f'Learning curve with alpha = {alpha}')
            plt.plot(it, losses)
            plt.xlabel('number of iteration')
            plt.ylim([min(losses) * 1.00001, max(losses) * 1.00001])
            plt.ylabel('Loss')
            plt.show()

            # predict and add polynomial features for each test feature
            x_test_ = polynomial_matrix(x_test, degree)
            y_hat_test = my_lr.predict_(x_test_)

            # Evaluates and prints evaluation score (loss) of each model
            loss = my_lr.loss_(y_test, y_hat_test)

            # saving parameters and results in the models.csv file
            save_training(writer, nb_model, degree, my_lr.thetas, alpha,
                            max_iter, loss)

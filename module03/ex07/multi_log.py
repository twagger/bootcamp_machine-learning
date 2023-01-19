"""
Multi log program :

1. Split the dataset into a training and a test set.

2. Train 4 logistic regression classifiers to discriminate each class from the
    others (the way you did in part one).

3. Predict for each example the class according to each classifiers and select
    the one with the highest output probability.

4. Calculate and display the fraction of correct predictions over the total
    number of predictions based on the test set.

5. Plot 3 scatter plots (one for each pair of citizen features) with the
    dataset and the final prediction of the model.

"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Helper functions
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
        # type test
        if not isinstance(proportion, float):
            print('Something went wrong', file=sys.stderr)
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


# normalization functions
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
            print("Something went wrong", file=sys.stderr)
            return None
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


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 1. Data loading and preparation (no cleaning needed here)
    # -------------------------------------------------------------------------
    try:
        df_features = pd.read_csv("./solar_system_census.csv")
        df_labels = pd.read_csv("./solar_system_census_planets.csv")
    except:
        # At least to catch :
        # FileNotFoundError, PermissionError, pandas.errors.ParserError,
        # pandas.errors.DtypeWarning, pandas.errors.EmptyDataError,
        # pandas.errors.ParserWarning, UnicodeDecodeError:
        print("Error when trying to read dataset", file=sys.stderr)
        sys.exit()

    # concatenate the data into one dataframe
    df = pd.concat([df_features[['weight', 'height', 'bone_density']],
                    df_labels['Origin']], axis=1)

    # rename the columns to ease the next manipulations
    df = df.rename(columns={'weight':'w', 'height':'h', 'bone_density':'d',
                            'Origin':'o'})

    # -------------------------------------------------------------------------
    # 2. Training set and Test set
    # -------------------------------------------------------------------------
    # split the dataset into a training set and a test set
    X = np.array(df[['w', 'h', 'd']]).reshape((-1, 3))
    # normalization : not very efficient here
    # X = normalize_xset(X)
    y = np.array(df['o']).reshape((-1, 1))
    X_train, X_test, y_train, y_test = data_spliter(X, y, 0.8)

    # create train and test label vectors with only 2 possibilities
    y_trains = []
    y_tests = []
    for zipcode in range(4):
        relabel_log = np.vectorize(lambda x: 1 if x == zipcode else 0)
        y_trains.append(relabel_log(y_train))
        y_tests.append(relabel_log(y_test))

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

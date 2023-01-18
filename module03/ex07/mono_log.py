"""
Mono log program :

1. Take an argument: -zipcode=x with x being 0, 1, 2 or 3.
    If no argument, usage will be displayed.

2. Split the dataset into a training and a test set.

3. Select your favorite Space Zipcode and generate a new numpy.array to label
    each citizen according to your new selection criterion:
    • 1 if the citizen's zipcode corresponds to your favorite planet.
    • 0 if the citizen has another zipcode.

4. Train a logistic model to predict if a citizen comes from your favorite
    planet or not, using your brand new label.

5. Calculate and display the fraction of correct predictions over the total
    number of predictions based on the test set.

6. Plot 3 scatter plots (one for each pair of citizen features) with the
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


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 1. Parsing arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-zipcode",
                        help="zipcode for classification (0, 1, 2, 3)",
                        required=True, type=int)
    zipcode = parser.parse_args().zipcode
    if zipcode not in range(4):
        print("Range error on argument", file=sys.stderr)
        sys.exit()

    # -------------------------------------------------------------------------
    # 2. Data loading and preparation (no cleaning needed here)
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
    # 3. Training set and Test set
    # -------------------------------------------------------------------------
    # split the dataset into a training set and a test set
    X = np.array(df[['w', 'h', 'd']]).reshape((-1, 3))
    y = np.array(df['o']).reshape((-1, 1))
    X_train, X_test, y_train, y_test = data_spliter(X, y, 0.8)

    # create train and test label vectors with only 2 possibilities
    relabel_log = np.vectorize(lambda x: 1 if x == zipcode else 0)
    y_train_log = relabel_log(y_train)
    y_test_log = relabel_log(y_test)

    # -------------------------------------------------------------------------
    # 4. Model training
    # -------------------------------------------------------------------------
    # train a logistic model to predict if the citizen is from the zipcode or
    # not
    sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex06'))
    from my_logistic_regression import MyLogisticRegression as MyLR

    # create model
    myLR = MyLR(np.random.rand(4, 1), alpha=1e-5, max_iter=100000)
    # evaluate model before training (loss)
    print(f'{"Loss of the model before training ":33}: '
          f'{myLR.loss_(y_test_log, myLR.predict_(X_test))}')
    # train model
    myLR.fit_(X_train, y_train_log)

    # -------------------------------------------------------------------------
    # 5. Model evaluation
    # -------------------------------------------------------------------------
    # evaluate model after training (loss)
    y_hat_test = myLR.predict_(X_test)
    print(f'{"Loss of the model after training ":33}: '
          f'{myLR.loss_(y_test_log, y_hat_test)}')

    # apply a 0.5 threshold on the predictions and reevaluate the loss
    threshold = np.vectorize(lambda x: 1 if x > 0.5 else 0)
    y_hat_test_01 = threshold(y_hat_test)
    print(f'{"Loss after threshold ":33}: '
          f'{myLR.loss_(y_test_log, y_hat_test_01)}')

    # fraction of correct predictions over the total number of predictions
    # based on the test set.
    nb_pred = len(y_hat_test)
    # correct_pred = nb_pred - np.sum(np.abs(y_test_log - y_hat_test_01))
    correct_pred = np.sum(y_test_log == y_hat_test_01)
    print(f'{"fraction of correct prediction ":33}: '
          f'{correct_pred / nb_pred}'
          f' ({correct_pred}/{nb_pred})')

    # -------------------------------------------------------------------------
    # 6. plot 3 scatter plots : prediction vs true citizenship per feature
    # -------------------------------------------------------------------------
    # relabel y in 2 categories
    y = relabel_log(y)
    # predict all data (train + test sets)
    y_hat = myLR.predict_(X)
    # threshold the predictions
    y_hat = threshold(y_hat)
    # plot
    plt.figure()
    plt.title('Prediction versus true citizenship')
    plt.ylabel('citizenship')
    plt.subplot(1,3,1)
    plt.xlabel('Weight')
    plt.scatter(np.array(df['w']).reshape((-1, 1)), y, label='True')
    plt.scatter(np.array(df['w']).reshape((-1, 1)), y_hat, label='Predicted',
                marker='x')
    plt.subplot(1,3,2)
    plt.xlabel('Height')
    plt.scatter(np.array(df['h']).reshape((-1, 1)), y, label='True')
    plt.scatter(np.array(df['h']).reshape((-1, 1)), y_hat, label='Predicted',
                marker='x')
    plt.subplot(1,3,3)
    plt.xlabel('Bone density')
    plt.scatter(np.array(df['d']).reshape((-1, 1)), y, label='True')
    plt.scatter(np.array(df['d']).reshape((-1, 1)), y_hat, label='Predicted',
                marker='x')
    plt.legend()
    plt.show()

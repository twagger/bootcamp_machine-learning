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
from ridge import type_validator, shape_validator
from my_logistic_regression import MyLogisticRegression as MyLogR


# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
# Global params
max_iter = 500000
alpha = 1e-1
cols = ['w', 'w2', 'w3', 'h', 'h2', 'h3', 'd', 'd2', 'd3']


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


# Saving to file function
@type_validator
@shape_validator({'thetas': ('n', 1)})
def save_training(writer, nb_model: int, form: str, thetas: np.ndarray,
                  alpha: float, max_iter: int, lambda_: float,
                  f1_score: float):
    """save the training in csv file"""
    try:
        thetas_str = ','.join([f'{theta[0]}' for theta in thetas])
        writer.writerow([nb_model, form, thetas_str, alpha, max_iter, lambda_,
                         f1_score])
    except:
        return None


# model training function to push it to multithreading
@type_validator
@shape_validator({'X_train': ('m', 'n'), 'Y_train': ('m', 1)})
def train_model(model: ModelWithInfos, X_train: np.ndarray,
                y_train: np.ndarray):
    """Train the model"""
    try:
        # Select only the subset of features associated with the model
        features = model.features
        f_indices = [i for i, feat in enumerate(cols) if feat in features]
        X_train = X_train.copy()[:, f_indices]
        # Train model
        model.m.fit_(X_train, y_train)
    except:
        return None


# normalization functions
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


# misc tools
@type_validator
def save_model(num: int, models: list, model: MyLogR, features: list):
    """save a model into a dictionnary of models"""
    models.append(ModelWithInfos(num=num, m=model, features=features,
                                 f1_score=None))

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
# helper function to calculate tp, fp, tn, fn
@type_validator
@shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
def tf_metrics(y: np.ndarray, y_hat: np.ndarray, pos_label=None) -> tuple:
    """
    Returns as a tuple in that order :
        true positive number
        false positive number
        true negative number
        false negative number
    """
    # initialize variables
    tp, fp, tn, fn = [0]*4
    # if global for all classes
    if pos_label==None:
        # loop on every class in the original y vector
        for class_ in np.unique(y):
            tp += np.sum(np.logical_and(y_hat == class_, y == class_))
            fp += np.sum(np.logical_and(y_hat == class_, y != class_))
            tn += np.sum(np.logical_and(y_hat != class_, y != class_))
            fn += np.sum(np.logical_and(y_hat != class_, y == class_))
    else: # focus on one class
        tp += np.sum(np.logical_and(y_hat == pos_label, y == pos_label))
        fp += np.sum(np.logical_and(y_hat == pos_label, y != pos_label))
        tn += np.sum(np.logical_and(y_hat != pos_label, y != pos_label))
        fn += np.sum(np.logical_and(y_hat != pos_label, y == pos_label))

    return (tp, fp, tn, fn)


def accuracy_score_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute the accuracy score : how many predictions are correct.
    """
    try:
        tp, fp, tn, fn = tf_metrics(y, y_hat)
        return (tp + tn) / (tp + fp + tn + fn)
    except:
        return None

@type_validator
@shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
def precision_score_(y: np.ndarray, y_hat: np.ndarray,
                     pos_label: int = 1) -> float:
    """
    Compute the precision score : model's ability to not classify positive
                                  examples as negative.
    """
    try:
        tp, fp, _, _ = tf_metrics(y, y_hat, pos_label)
        return tp / (tp + fp)
    except:
        return None


@type_validator
@shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
def recall_score_(y: np.ndarray, y_hat: np.ndarray,
                  pos_label: int = 1) -> float:
    """
    Compute the recall score : model's ability to detect positive examples.
    """
    try:
        tp, _, _, fn = tf_metrics(y, y_hat, pos_label)
        return tp / (tp + fn)
    except:
        return None


@type_validator
@shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
def f1_score_(y: np.ndarray, y_hat: np.ndarray,
              pos_label: int = 1) -> float:
    """
    Compute the f1 score : harmonic mean of precision and recall. often used
                           for imbalanced datasets where it is important to
                           minimize false negatives while minimizing false
                           positives.
    """
    try:
        precision = precision_score_(y, y_hat, pos_label=pos_label)
        recall = recall_score_(y, y_hat, pos_label=pos_label)
        return (2 * precision * recall) / (precision + recall)
    except:
        return None


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
    # add polynomial features up to degree 3 : w, w2, w3, h, h2, h3, d, d2, d3
    X_poly = polynomial_matrix(X, 3)

    # normalize to ease gradient descent
    X_norm = normalize_xset(X_poly)

    # switch back to dataframe and relabel columns to ease feature selection
    # during model training
    df = pd.DataFrame(data=X_norm, columns=cols)

    # -------------------------------------------------------------------------
    # 3. Split data : training set / cross validation set / test set (60/20/20)
    # -------------------------------------------------------------------------
    # Split dataset : train / cross validation / test (60/20/20)
    # train / validation + test (60/40)
    X_train, X_val_test, y_train, y_val_test = data_spliter(X_norm, y, 0.6)
    # cross validation / test (50/50)
    X_test, X_val, y_test, y_val = data_spliter(X_val_test, y_val_test, 0.5)

    # -------------------------------------------------------------------------
    # 3. Create train and cross validation label vectors with only 2 outputs
    # -------------------------------------------------------------------------
    y_trains = []
    y_vals = []
    for zipcode in range(4):
        relabel_log = np.vectorize(lambda x: 1 if x == zipcode else 0)
        y_trains.append(relabel_log(y_train))
        y_vals.append(relabel_log(y_val))

    # -------------------------------------------------------------------------
    # 4. Create different regularized logistic regression models with a
    #    polynomial hypothesis of degree 3.
    #    Create the models with different λ values: from 0 to 1 (0.2 step)
    # -------------------------------------------------------------------------
    models = []
    # apply λ 0 to 1 (with 0.2 step)
    for num, lambda_ in enumerate(np.arange(0.0, 1.2, 0.2)):

        # train 4 different models per λ value as we have 4 classes to predict

        # degree 1
        for _ in range(4):
            model = MyLogR(np.random.rand(4, 1), alpha = alpha,
                           max_iter = max_iter, lambda_ = lambda_)
            save_model(1 + num, models, model, ['w',
                                                'h',
                                                'd'])
        # degree 2
        for _ in range(4):
            model = MyLogR(np.random.rand(7, 1), alpha = alpha,
                           max_iter = max_iter, lambda_ = lambda_)
            save_model(100 + num, models, model, ['w',
                                                  'w2',
                                                  'h',
                                                  'h2',
                                                  'd',
                                                  'd2'])
        # degree 3
        for _ in range(4):
            model = MyLogR(np.random.rand(10, 1), alpha = alpha,
                           max_iter = max_iter, lambda_ = lambda_)
            save_model(10000 + num, models, model, ['w',
                                                    'w2',
                                                    'w3',
                                                    'h',
                                                    'h2',
                                                    'h3',
                                                    'd',
                                                    'd2',
                                                    'd3'])

    # sort the models to regroup then by degree
    models = sorted(models, key=lambda x: x.num)

    # -------------------------------------------------------------------------
    # 5. Train models
    # -------------------------------------------------------------------------
    # model training with modified outputs (0 or 1 only)
    for num, model in enumerate(tqdm(models)):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(train_model, model, X_train, y_trains[num % 4])

    # -------------------------------------------------------------------------
    # 6. Evaluate models with F1-score on the cross validation set
    # -------------------------------------------------------------------------
    for i in range(0, len(models), 4):
        # adapt features of X_val
        features = models[i].features
        f_indices = [i for i, feat in enumerate(cols) if feat in features]
        X_val_tmp = X_val.copy()[:, f_indices]
        # build a prediction vector for each group of 4 models
        predict = np.empty((y_val.shape[0], 0))
        for num in range(4):
            predict = np.c_[predict, models[i + num].m.predict_(X_val_tmp)]
        predict = np.argmax(predict, axis=1).reshape((-1, 1))
        # compute the f1-score for the group
        multiclass_f1_score: float = 0.0
        for label in range(4):
            multiclass_f1_score += f1_score_(predict, y_val, pos_label = label)
        # put the global f1-score on every sub-model
        for num in range(4):
            models[i + num].f1_score = multiclass_f1_score

    # -------------------------------------------------------------------------
    # 7. Save all models with their hyperparameters and results
    # -------------------------------------------------------------------------
    # open csv file to save params
    with open('models.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "features", "thetas", "alpha", "max_iter",
                         "lambda", "f1_score"])

        for ii, model in enumerate(models):
            # saving parameters and results in the models.csv file
            feat_str = ",".join([f'{feat}' for feat in model.features])
            save_training(writer, ii, feat_str, model.m.thetas, alpha,
                          max_iter, model.m.lambda_, model.f1_score)

    # -------------------------------------------------------------------------
    # 8. Pick the best model num and copy it at the end so it can be used after
    # -------------------------------------------------------------------------
    # pick the model with the highest f1 score
    the_model = max(models, key=lambda x: x.f1_score)
    # aggregate all 4 models in a list
    the_models = []
    for model in models:
        if model.num == the_model.num:
            the_models.append(model)

    # save it as last with its 3 mates in the models.csv file to be retrieved
    with open('models.csv', 'a') as file:
        writer = csv.writer(file)
        for the_model in the_models:
            feat_str = ",".join([f'{feat}' for feat in the_model.features])
            save_training(writer, 9999, feat_str, the_model.m.thetas,
                          alpha, max_iter, the_model.m.lambda_,
                          the_model.f1_score)

    # -------------------------------------------------------------------------
    # 9. Evaluate: fraction of correct predictions over the total number of
    #              predictions based on the test set.
    # -------------------------------------------------------------------------
    # here we cannot use the loss function to evaluate the global loss as it
    # expect prediction between 0 and 1

    # Select only the subset of features associated with the best model
    features = the_model.features
    f_indices = [i for i, feat in enumerate(cols) if feat in features]
    X_test = X_test[:, f_indices]

    # prediction with the best group of models on test set
    predict = np.empty((y_test.shape[0], 0))
    for mod in the_models:
        predict = np.c_[predict, mod.m.predict_(X_test)]
    predict = np.argmax(predict, axis=1).reshape((-1, 1))

    # fraction of correct predictions over the total number of predictions
    # based on the test set.
    nb_pred = len(predict)
    correct_pred = np.sum(y_test == predict)
    print(f'{"fraction of correct prediction ":33}: '
          f'{correct_pred / nb_pred}'
          f' ({correct_pred}/{nb_pred})')

    # -------------------------------------------------------------------------
    # 10. plot f1 score per lambda
    # -------------------------------------------------------------------------
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
    # 11. plot 3 scatter plots : prediction vs true citizenship per feature
    # -------------------------------------------------------------------------
    # plot
    plt.figure()
    plt.title('Prediction versus true citizenship')
    plt.ylabel('citizenship')
    plt.subplot(1,3,1)
    plt.xlabel('Weight')
    plt.scatter(X_test[:, features.index('w')], y_test, label='True')
    plt.scatter(X_test[:, features.index('w')], predict, label='Predicted',
                marker='x')
    plt.subplot(1,3,2)
    plt.xlabel('Height')
    plt.scatter(X_test[:, features.index('h')], y_test, label='True')
    plt.scatter(X_test[:, features.index('h')], predict, label='Predicted',
                marker='x')
    plt.subplot(1,3,3)
    plt.xlabel('Bone density')
    plt.scatter(X_test[:, features.index('d')], y_test, label='True')
    plt.scatter(X_test[:, features.index('d')], predict, label='Predicted',
                marker='x')
    plt.legend()
    plt.show()

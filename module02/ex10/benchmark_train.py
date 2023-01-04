"""
Benchmark train module
This program performs the training of all the models and save the parameters
of the different models into a file.
In models.csv are the parameters of all the models I have explored and trained.
"""
import numpy as np
import panda as pd
import csv


if __name__ == "__main__":

    # import necessary modules
    import sys
    sys.path.insert(1, '../ex09/')
    from data_spliter import data_spliter
    sys.path.insert(1, '../ex07/')
    from polynomial_model import add_polynomial_features
    sys.path.insert(1, '../ex05')
    from mylinearregression import MyLinearRegression as MyLR

    # read the dataset
    dataset = pd.read_csv("./space_avocado.csv")

    # open csv file to save params
    with open('models.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "degree", "thetas", "alpha", "max_iter",
                         "split"])

        # create multiple training set and a test set to optimize the chance of
        # having a good result
        nb_model = 0
        for split in range(1, 4):

            X = np.array(dataset['weight', 'prod_distance',
                                'time_delivery']).reshape((-1, 3))
            y = np.array(dataset['target']).reshape((-1, 1))
            x_train, x_test, y_train, y_test = data_spliter(X, y, split * 0.2)

            # Trains 4 separate Linear Regression models with polynomial
            # hypothesis with degrees ranging from 1 to 4
            for degree in range(1, 5):

                # global params
                theta = np.ones(degree + 1).reshape(-1,1)
                alpha = 2.5e-5
                max_iter = 500000
                # specific params
                if degree == 10:
                    theta = np.array([[-20.0],[ 160],[ -80],[ 10],
                                    [ -1]]).reshape(-1,1)
                    alpha = 2.5e-8
                    max_iter = 500000

                # add polynomial data
                x_ = add_polynomial_features(X, degree)
                # create the model
                my_lr = MyLR(theta, alpha = alpha, max_iter = max_iter)
                nb_model += 1
                # fit the model
                my_lr.fit_(x_, y)
                # predict
                y_hat = my_lr.predict_(x_)
                # Evaluates and prints evaluation score (MSE) of each model
                mse = my_lr.mse_(y, y_hat)
                # saving parameters and results in the models.csv file
                f_thetas = '|'.join(str(t) for t in my_lr.thetas.flatten())
                writer.writerow([nb_model,
                                 degree,
                                 f_thetas,
                                 alpha,
                                 max_iter,
                                 split * 0.2])

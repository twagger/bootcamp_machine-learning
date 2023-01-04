"""Multivariate linear model"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Class copied from module 1 to perform univariate linear regression
class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000):
        """Constructor"""
        self.alpha = alpha
        self.max_iter = max_iter
        try:
            self.thetas = np.array(thetas).reshape((-1, 1))
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc)

    def predict_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x and/or theta are not numpy.array.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            # shape
            x = x.reshape((-1, 1))
            # creation of the prediction matrix
            x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
            return x_prime.dot(self.thetas)
        except (TypeError, ValueError, AttributeError) as exc:
            print(exc)
            return None

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_elem: numpy.array, a vector of dimension (number of the training
                examples,1).
            None if there is a dimension matching problem between X, Y or
                theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # shapes
            y = y.reshape((-1, 1))
            y_hat = y_hat.reshape((-1, 1))
            # calculation
            return (y_hat - y) ** 2
        except (ValueError, TypeError, AttributeError):
            return None

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Computes the half mean squared error of two non-empty numpy.array,
        without any for loop.
        The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            The half mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.array.
            None if y and y_hat does not share the same dimensions.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            loss_vector = self.loss_elem_(y, y_hat)
            return float(np.sum(loss_vector) / (2 * len(y)))
        except (ValueError, TypeError, AttributeError):
            return None

    def mse_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Computes the mean squared error of two non-empty numpy.array,
        without any for loop.
        The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            The half mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.array.
            None if y and y_hat does not share the same dimensions.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
           loss_vector = self.loss_elem_(y, y_hat)
           return float(np.sum(loss_vector) / len(y))
        except (ValueError, TypeError, AttributeError):
            return None

    def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
        Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # shapes
            x = x.reshape((-1, 1))
            y = y.reshape((-1, 1))
            # 0. add intercept
            m = x.shape[0]
            x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
            # 1. loop
            for _ in range(self.max_iter):
                # 2. calculate the gradient for current thetas
                y_hat = self.predict_(x)
                gradient = x_prime.T.dot(y_hat - y) / m
                # 3. calculate and assign the new thetas
                self.thetas[0][0] -= self.alpha * gradient[0][0]
                self.thetas[1][0] -= self.alpha * gradient[1][0]
            return self.thetas
        except (ValueError, TypeError, AttributeError):
            return None

# end of univariate linear regression class

if __name__ == "__main__":

    # read data
    data = pd.read_csv("./spacecraft_data.csv")

    # UNIVARIATE

    # select a feature to predict Sell price
    for feature in data.head(0).iloc[:, :-1]:
        X = np.array(data[[feature]])
        Y = np.array(data[['Sell_price']])

        # change the initial thetas according to the feature
        params = {'Age': {'thetas': [[1000.0], [-1.0]],
                          'colors' : ['darkblue', 'dodgerblue'],
                          'legend': 'x₁: age (in years)'},
                  'Thrust_power':  {'thetas': [[-1.0], [1000.0]],
                          'colors' : ['darkgreen', 'lime'],
                          'legend': 'x₂: thrust power (in 10Km/s)'},
                  'Terameters':  {'thetas': [[780.0], [-1.0]],
                          'colors' : ['darkviolet', 'violet'],
                          'legend': 'x₃: terameters (in Tmeters)'}}

        # create a first model for the 'Age' feature
        myLR_feat = MyLinearRegression(thetas = params[feature]['thetas'],
                                       alpha = 2.5e-5, max_iter = 100000)

        # fit to adjust the thetas
        myLR_feat.fit_(X[:,0].reshape(-1,1), Y)

        # predict with adjusted thetas
        y_hat = myLR_feat.predict_(X[:,0].reshape(-1,1))

        # output a scatter plot
        plt.figure()
        plt.grid()
        plt.title(f'Prediction of spacecrafts sell price with respect to their'
                  f' {feature.lower()}')
        plt.scatter(X, Y, marker='o', color=params[feature]['colors'][0],
                    label=feature)
        plt.scatter(X, y_hat, marker='o', color=params[feature]['colors'][1],
                    label=f'predicted {feature}')
        plt.xlabel(params[feature]['legend'])
        plt.ylabel('y: sell price (in keuros)')
        plt.legend()
        plt.show()

        # calculate the mean squared error
        print(f'Final loss for {feature}: {myLR_feat.mse_(y_hat, Y)}')
        print(f'Final thetas for {feature}:\n{myLR_feat.thetas}\n')

        # Output
        # 55736.86719... > This is a big error :(


    # MULTIVARIATE
    
    # import class
    import sys
    sys.path.insert(1, '../ex05/')
    from mylinearregression import MyLinearRegression as mlr_multi

    # X is all feature but the one we try to predict
    Y = np.array(data[['Sell_price']])
    X = np.array(data[['Age','Thrust_power','Terameters']])

    features = list(data.head(0))[:-1]

    # create a first model for the 'Age' feature
    myLR = mlr_multi(thetas = [1.0, 1.0, 1.0, 1.0], alpha = 2.5e-5,
                     max_iter = 1000000)

    # fit to adjust the thetas
    myLR.fit_(X, Y)

    # predict with adjusted thetas
    y_hat = myLR.predict_(X)

    # prepare params for plotting
    params = {'Age': {'colors' : ['darkblue', 'dodgerblue'],
                      'legend': 'x₁: age (in years)'},
              'Thrust_power':  {'colors' : ['darkgreen', 'lime'],
                                'legend': 'x₂: thrust power (in 10Km/s)'},
              'Terameters':  {'colors' : ['darkviolet', 'violet'],
                              'legend': 'x₃: terameters (in Tmeters)'}}

    # plotting the predictions
    for indx, feature in enumerate(features):
        plt.figure()
        plt.grid()
        plt.title(f'Prediction of spacecrafts sell price with respect to their'
                  f' {feature.lower()}')
        plt.scatter(X[:, indx], Y, marker='o',
                    color=params[feature]['colors'][0],
                    label=feature)
        plt.scatter(X[:, indx], y_hat, marker='o', s=20,
                    color=params[feature]['colors'][1],
                    label=f'predicted {feature}')
        plt.xlabel(params[feature]['legend'])
        plt.ylabel('y: sell price (in keuros)')
        plt.legend()
        plt.show()

    # calculate the mean squared error
    print(f'Final loss : {myLR.mse_(y_hat, Y)}') # <- should be way better

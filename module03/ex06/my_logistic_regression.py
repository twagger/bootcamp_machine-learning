"""My Logistic Regression module"""
import math
import sys
import numpy as np
import pandas as pd
sys.path.insert(1, '../ex00/')
from sigmoid import sigmoid_


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000):
        if not isinstance(alpha, float) or not isinstance(max_iter, int):
            print('Something went wrong', file=sys.stderr)
            return
        self.alpha = alpha
        self.max_iter = max_iter
        try:
            self.theta = np.array(theta).reshape((-1, 1))
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)

    def predict_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the vector of prediction y_hat from two non-empty
        numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a vector of dimension m * n.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # shape test
            m, n = x.shape
            if (self.theta.shape[0] != n + 1 or self.theta.shape[1] != 1):
                print('Something went wrong', file=sys.stderr)
                return None
            # calculation
            x_prime = np.c_[np.ones((m, 1)), x]
            return sigmoid_(x_prime.dot(self.theta))
        except (TypeError, ValueError, AttributeError) as exc:
            print(exc, file=sys.stderr)
            return None

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Description:
        Calculates the loss by element.
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
            # shape test
            if (y.shape[0] != y_hat.shape[0] or y.shape[1] != 1
                    or y_hat.shape[1] != 1):
                print('Something went wrong', file=sys.stderr)
                return None
            # values check
            min_val = 0
            max_val = 1
            valid_values = np.logical_and(min_val <= y_hat, y_hat <= max_val)
            if not np.all(valid_values):
                print('Something went wrong', file=sys.stderr)
                return None
            # add a little value to y_hat to avoid log(0) problem
            eps: float=1e-15
            # y_hat = np.clip(y_hat, eps, 1 - eps) < good options for eps
            return -(y * np.log(y_hat + eps) + (1 - y) \
                    * np.log(1 - y_hat + eps))
        except (TypeError, ValueError, AttributeError) as exc:
            print(exc, file=sys.stderr)
            return None

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Compute the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
            The logistic loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        try:
            m, _= y.shape
            return np.sum(self.loss_elem_(y, y_hat)) / m
        except (TypeError, ValueError, AttributeError) as exc:
            print(exc, file=sys.stderr)
            return None

    def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * n:
                (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
        Returns:
            new_theta: numpy.ndarray, a vector of dimension (n + 1) * 1.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # shape test
            if (x.shape[0] != y.shape[0] or y.shape[1] != 1):
                print('Something went wrong', file=sys.stderr)
                return None
            # calculation of the gradient vector
            # 1. X to X'
            m = x.shape[0]
            x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
            # 2. loop
            for _ in range(self.max_iter):
                # 3. calculate the grandient for current thetas
                y_hat = self.predict_(x)
                gradient = x_prime.T.dot(y_hat - y) / m
                # 4. calculate and assign the new thetas
                self.theta -= self.alpha * gradient
            return self.theta
        except (TypeError, ValueError, AttributeError) as exc:
            print(exc, file=sys.stderr)
            return None


if __name__ == "__main__":

    # setting parameters
    X = np.array([[1., 1., 2., 3.],
                  [5., 8., 13., 21.],
                  [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLogisticRegression(thetas)

    print(f'dataset Y:\n{Y}\n')

    # Example 0:
    y_hat = mylr.predict_(X)
    print(f'firt predict:\n{y_hat}\n')
    # Output:
    # array([[0.99930437],
    #        [1. ],
    #        [1. ]])

    # Example 1:
    print(f'loss:\n{mylr.loss_(Y, y_hat)}\n')
    print(f'loss by elem:\n{mylr.loss_elem_(Y, y_hat)}\n')
    # Output:
    # 11.513157421577004

    # Example 2:
    mylr.fit_(X, Y)
    print(f'fitted thetas:\n{mylr.theta}\n')
    # Output:
    # array([[ 2.11826435]
    #        [ 0.10154334]
    #        [ 6.43942899]
    #        [-5.10817488]
    #        [ 0.6212541 ]])

    # Example 3:
    y_hat = mylr.predict_(X)
    print(f'new prediction:\n{y_hat}\n')
    # Output:
    # array([[0.57606717]
    #        [0.68599807]
    #        [0.06562156]])

    # Example 4:
    print(f'new loss:\n{mylr.loss_(Y, y_hat)}\n')
    # Output:
    # 1.4779126923052268

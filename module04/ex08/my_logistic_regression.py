"""My Logistic Regression module"""
import math
import sys
import numpy as np
import pandas as pd
import inspect
from functools import wraps


# -----------------------------------------------------------------------------
# decorators
# -----------------------------------------------------------------------------
# generic type validation based on type annotation in function prototype
def type_validator(func):
    # extract information about the function's parameters and return type.
    sig = inspect.signature(func)
    # preserve name and docstring of decorated function
    @wraps(func)
    def wrapper(*args, **kwargs):
        # map the parameter names to their corresponding values
        bound_args = sig.bind(*args, **kwargs)
        # check for each name of param if value has the declared type
        for name, value in bound_args.arguments.items():
            if name in sig.parameters:
                param = sig.parameters[name]
                if (param.annotation != param.empty
                        and not isinstance(value, param.annotation)):
                    print(f"Expected type '{param.annotation}' for argument " \
                          f"'{name}' but got {type(value)}.")
                    return None
        return func(*args, **kwargs)
    return wrapper

# shape validation (a bit long and not very elegant. I should redo it)
def shape_validator(shape_mapping: dict):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            m_value: int = None
            n_value = self.theta.shape[0] - 1 \
                      if isinstance(self, MyLogisticRegression) else None
            for nparray_name, expected_shape in shape_mapping.items():
                if hasattr(self, nparray_name):
                    arr = getattr(self, nparray_name)

                    # type and dim check to avoid possible exception later
                    if not isinstance(arr, np.ndarray):
                        print(f'{nparray_name} must be a numpy array',
                              file=sys.stderr)
                        return None
                    if arr.ndim != 2:
                        print(f'{nparray_name} must be 2D array',
                              file=sys.stderr)
                        return None

                    # type and value checks on expected shape
                    if not (isinstance(expected_shape, tuple)
                            and len(expected_shape) != 2):
                        print(f'{expected_shape} must be a tuple of 2 values',
                              file=sys.stderr)
                        return None

                    # shape test
                    m, n = expected_shape
                    # Generic check
                    if expected_shape == arr.shape:
                        continue
                    # Specific dimensions
                    if m == 'm':
                        if m_value is None:
                            m_value = arr.shape[0]
                        elif m_value != arr.shape[0]:
                            print(f'{nparray_name} has an invalid shape on ' \
                                  f'dimension 0. Expected {m_value}, got ' \
                                  f'{arr.shape[0]}.')
                            return None
                    if n == 'n':
                        if n_value is None:
                            n_value = arr.shape[1]
                        elif n_value != arr.shape[1]:
                            print(f'{nparray_name} has an invalid shape on ' \
                                  f'dimension 1. Expected {n_value}, got ' \
                                  f'{arr.shape[1]}.')
                            return None
                    if m == 'n + 1':
                        if n_value is None:
                            print(f'wrong parameter order. ' \
                                  f'You must define n before n + 1.')
                            return None
                        elif n_value + 1 != arr.shape[0]:
                            print(f'{nparray_name} has an invalid shape on ' \
                                  f'dimension 0. Expected {n_value}, got ' \
                                  f'{arr.shape[0]}.')
                            return None
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# regulatization of loss
def regularize_loss(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        y, y_hat = args
        m, _ = y.shape
        loss = func(self, y, y_hat)
        regularization_term = (self.lambda_ / (2 * m)) * l2(theta)
        return loss + regularization_term
    return wrapper


# regulatization of gradient
def regularize_grad(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        x, y = args
        m, _ = x.shape
        gradient = func(self, x, y)
        # add regularization
        theta_prime = self.thetas.copy()
        theta_prime[0][0] = 0
        return gradient + (self.lambda_ * theta_prime) / m
    return wrapper


# -----------------------------------------------------------------------------
# helper function
# -----------------------------------------------------------------------------
# sigmoid
@type_validator
@shape_validator({'x': ('m', 1)})
def sigmoid_(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    try:
        return 1 / (1 + np.exp(-x))
    except:
        return None


# L2 regularization
@type_validator
def l2(theta: np.ndarray) -> float:
    """
    Computes the L2 regularization of a non-empty numpy.ndarray, without any
        for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # type test
        if not isinstance(theta, np.ndarray):
            print('thetas must be a numpy array', file=sys.stderr)
            return None
        # shape test
        if theta.shape[1] != 1:
            print('wrong shape on parameter', file=sys.stderr)
            return None
        # l2
        theta_prime = theta
        theta_prime[0][0] = 0
        return theta_prime.T.dot(theta_prime)[0][0]

    except:
        return None

# -----------------------------------------------------------------------------
# MyLogisticRegression class with l2 regularization
# -----------------------------------------------------------------------------
class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    # We consider l2 penalty only. One may wants to implement other penalties
    supported_penalties = ['l2']

    @type_validator
    def __init__(self, theta: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000, penalty: str = 'l2',
                 lambda_: float = 1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.array(theta).reshape((-1, 1))
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalties else 0

    @type_validator
    @shape_validator({'x': ('m', 'n')})
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
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]
            return sigmoid_(x_prime.dot(self.theta))
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
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
            # values check
            min_val = 0
            max_val = 1
            valid_values = np.logical_and(min_val <= y_hat, y_hat <= max_val)
            if not np.all(valid_values):
                print('y / y_hat val must be between 0 and 1', file=sys.stderr)
                return None
            # add a little value to y_hat to avoid log(0) problem
            eps: float=1e-15
            # y_hat = np.clip(y_hat, eps, 1 - eps) < good options for eps
            return -(y * np.log(y_hat + eps) + (1 - y) \
                    * np.log(1 - y_hat + eps))
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    @regularize_loss
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
        except:
            return None

    @type_validator
    @shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
    @regularize_grad
    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the regularized linear gradient of three non-empty
        numpy.ndarray. The three arrays must have compatible shapes.
        Args:
            x: has to be a numpy.ndarray, a matrix of dimesion m * n.
            y: has to be a numpy.ndarray, a vector of shape m * 1.
        Return:
            A numpy.ndarray, a vector of shape (n + 1) * 1, containing the
                results of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles shapes.
            None if y, x or theta or lambda_ is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]
            y_hat = x_prime.dot(self.theta)
            return x_prime.T.dot(y_hat - y) / m
        except:
            return None

    @type_validator
    @shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
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
            # calculation of the gradient vector
            # 1. X to X'
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]
            # 2. loop
            for _ in range(self.max_iter):
                # 3. calculate the grandient for current thetas
                y_hat = self.predict_(x)
                gradient = self.gradient_(x, y)
                # 4. calculate and assign the new thetas
                self.theta -= self.alpha * gradient
            return self.theta
        except:
            return None


if __name__ == "__main__":

    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    # Example 1:
    model1 = MyLogisticRegression(theta, lambda_=5.0)
    print(model1.penalty) # -> 'l2'
    print(model1.lambda_) # -> 5.0

    # Example 2:
    model2 = MyLogisticRegression(theta, penalty='none')
    print(model2.penalty) # -> None
    print(model2.lambda_) # -> 0.0

    # Example 3:
    model3 = MyLogisticRegression(theta, penalty='none', lambda_=2.0)
    print(model3.penalty) # -> None
    print(model3.lambda_) # -> 0.0

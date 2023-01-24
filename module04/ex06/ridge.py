"""
Ridge regression :  linear regression regularized with L2
"""
import sys
import inspect
import numpy as np
from functools import wraps


# -----------------------------------------------------------------------------
# decorators
# -----------------------------------------------------------------------------
# generic type validation based on type annotation in function signature
def type_validator(func):
    # extract information about the function's parameters and return type.
    sig = inspect.signature(func)
    # preserve name and docstring of decorated function
    @wraps(func)
    def wrapper(*args, **kwargs):
        # map the parameter from signature to their corresponding values
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

# regulatization of loss
def regularize_loss(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        y, y_hat = args
        m, _ = y.shape
        loss = func(self, y, y_hat)
        theta_prime = self.thetas.copy()
        theta_prime[0][0] = 0
        regularization_term = (self.lambda_ / (2 * m)) * l2(theta_prime)
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
# helper functions
# -----------------------------------------------------------------------------
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


# my Linear regression class from previous module for inheritance
# normaly I should import this from another file but the files we can return in
# this exercise are limited to this one only so I copied the code directly in 
# it
class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    @type_validator
    def __init__(self, thetas: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000):
        """Constructor"""
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas).reshape((-1, 1))

    @type_validator
    def predict_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * n.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x and/or theta are not numpy.array.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            # shape test
            m, n = x.shape
            if self.thetas.shape[0] != n + 1:
                print('wrong shape on parameter', file=sys.stderr)
                return None
            x_prime = np.c_[np.ones((m, 1)), x]
            return x_prime.dot(self.thetas)
        except:
            return None

    @type_validator    
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
            y = y.reshape((-1, 1))
            y_hat = y_hat.reshape((-1, 1))
            return (y_hat - y) ** 2
        except:
            return None

    @type_validator
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
            m, _ = y.shape
            return float((np.sum(loss_vector) / (2 * m)))
        except:
            return None

    @type_validator
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
            # shape test
            m, _ = x.shape
            if y.shape[0] != m or y.shape[1] != 1:
                print("wrong shape on parameter", file=sys.stderr)
            # computation
            x_prime = np.c_[np.ones((m, 1)), x]
            y_hat = x_prime.dot(self.thetas)
            return x_prime.T.dot(y_hat - y) / m

        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)
            return None

    @type_validator
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
            m, _ = x.shape
            if (y.shape[0] != m or y.shape[1] != 1):
                print('wrong shape on parameter', file=sys.stderr)
                return None
            # calculation of the gradient vector
            # 1. X to X'
            x_prime = np.c_[np.ones((m, 1)), x]
            # 2. loop
            for _ in range(self.max_iter):
                # 3. calculate the grandient for current thetas
                gradient = self.gradient_(x, y)
                # 4. calculate and assign the new thetas
                self.thetas -= self.alpha * gradient
            return self.thetas

        except:
            return None


class MyRidge(MyLinearRegression):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """

    @type_validator
    def __init__(self, thetas: np.ndarray = None, alpha: float = 0.001,
                 max_iter: int = 1000, lambda_: float = 0.5):
        super().__init__(thetas, alpha=alpha, max_iter=max_iter)
        self.lambda_ = lambda_

    def get_params_(self) -> tuple:
        """Returns a tuple with the parameters of the model"""
        return (self.alpha, self.max_iter, self.lambda_, self.thetas)

    @type_validator
    def set_params_(self, thetas: np.ndarray = None, alpha: float = None,
                    max_iter: int = None, lambda_: float = None):
        """Update the parameters of the model"""
        if alpha is not None:
            self.alpha = alpha
        if max_iter is not None:
            self.max_iter = max_iter
        if lambda_ is not None:
            self.lambda_ = lambda_
        if thetas is not None:
            self.thetas = np.array(thetas).reshape((-1, 1))

    @type_validator
    @regularize_loss
    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return super().loss_(y, y_hat)

    @type_validator
    @regularize_grad
    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return super().gradient_(x, y)

# this module is tested in next exercise

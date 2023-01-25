"""
Ridge regression :  linear regression regularized with L2
"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
# nd arrays
import numpy as np
# for decorators
import inspect
from functools import wraps
# user modules
# no user modules here as it makes it complicated to use this one to import
# decorators in all others modules

# -----------------------------------------------------------------------------
# decorators
# -----------------------------------------------------------------------------
# generic type validation based on type annotation in function's signature
def type_validator(func):
    """
    Decorator that will rely on the types and attributes declaration in the
    function's signature to check the actual types of the parameter against the
    expected types
    """
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
                    print(f"function '{func.__name__}' : " \
                          f"expected type '{param.annotation}' for argument " \
                          f"'{name}' but got {type(value)}.")
                    return None
        return func(*args, **kwargs)
    return wrapper

# generic nd array shape validation based on function's signature
def shape_validator(shape_mapping: dict):
    """
    Decorator that will loop on the attributes in function signature and for
    each one checks if a specific 2D shape is expected in the dictionnary
    provided to the decorator.
    If the expected shape is not the current shape, the decorator prints an
    error and return None.
    This decorator does not do a lot of type checks, it must be verified before
    """
    def decorator(func):
        # extract information about the function's parameters.
        sig = inspect.signature(func)
        # preserve name and docstring of decorated function
        @wraps(func)
        def wrapper(*args, **kwargs):
            # init m and n so they can be used for comparison
            m_n: list = [None, None]
            # check positional arguments
            for i, (param_name, param) in enumerate(sig.parameters.items()):
                if param.annotation == np.ndarray and i < len(args):
                    arg = args[i]
                    expected_shape = shape_mapping.get(param_name)
                    # check the shape if there is something to check
                    if expected_shape is not None:
                        # dim check
                        if arg.ndim != 2:
                            print(f"function '{func.__name__}' : " \
                                  f"wrong dimension on '{param_name}'")
                            return None
                        # shape check
                        if not shape_ok(expected_shape, arg.shape, m_n):
                            print(f"function '{func.__name__}' : " \
                                  f"{param_name} has an invalid shape. "\
                                  f"Expected {expected_shape}, " \
                                  f"got {arg.shape}.")
                            return None

            # check keyword arguments
            for arg_name, expected_shape in shape_mapping.items():
                param = sig.parameters.get(arg_name)
                if param and param.annotation == np.ndarray:
                    arg = kwargs.get(arg_name)
                    if arg is not None:
                        # dim check
                        if arg.ndim != 2:
                            print(f"function '{func.__name__}' : " \
                                  f"wrong dimension on '{arg_name}'")
                            return None
                        # shape check
                        if not shape_ok(expected_shape, arg.shape, m_n):
                            print(f"function '{func.__name__}' : " \
                                  f"{arg_name} has an invalid shape. "\
                                  f"Expected {expected_shape}, " \
                                  f"got {arg.shape}.")
                            return None

            return func(*args, **kwargs)
        return wrapper
    return decorator


# shape decorator helper
def shape_ok(expected_shape: tuple, actual_shape: tuple, m_n: list) -> bool:
    """
    Helper to calculate if the expected shape matches the actual shape,
    taking m and n possibilities in account
    """
    m, n = expected_shape
    am, an = actual_shape
    # Generic check
    if expected_shape == actual_shape:
        return True
    # numeric dimensions
    if isinstance(m, int) and m != am:
        return False
    if isinstance(n, int) and n != an:
        return False
    # Specific dimensions 'm', 'n', 'n + 1'
    if m == 'm' and m_n[0] is not None and am != m_n[0]:
        return False
    if m == 'n' and m_n[1] is not None and am != m_n[1]:
        return False
    if n == 'n' and m_n[1] is not None and an != m_n[1]:
        return False
    if m == 'n + 1' and m_n[1] is not None and am != m_n[1] + 1:
        return False
    # if the param is the first with specific dimensions to be tested
    if m == 'm' and m_n[0] is None:
        m_n[0] = am
    if m == 'n' and m_n[1] is None:
        m_n[1] = am
    if n == 'n' and m_n[1] is None:
        m_n[1] = an
    return True


# regulatization of loss
def regularize_loss(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        y, y_hat = args
        m, _ = y.shape
        loss = func(self, y, y_hat)
        regularization_term = (self.lambda_ / (2 * m)) * l2(self.thetas)
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
@shape_validator({'theta': ('n', 1)})
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
        theta_prime = theta.copy()
        theta_prime[0][0] = 0
        return theta_prime.T.dot(theta_prime)[0][0]
    except:
        return None


# -----------------------------------------------------------------------------
# MyLinearRegression class : normaly I should import this from another file but
#                            the files we can return in this exercise are 
#                            limited to this one only so I copied the code 
#                            directly in it
# -----------------------------------------------------------------------------
class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    @type_validator
    @shape_validator({'thetas': ('n', 1)})
    def __init__(self, thetas: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000):
        """Constructor"""
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas).reshape((-1, 1))

    @type_validator
    @shape_validator({'x': ('m', 'n')})
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
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]
            return x_prime.dot(self.thetas)
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
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
            return (y_hat - y) ** 2
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
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
            m, _ = y.shape
            loss_vector = self.loss_elem_(y, y_hat)
            return float((np.sum(loss_vector) / (2 * m)))
        except:
            return None

    @type_validator
    @shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
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
            y_hat = x_prime.dot(self.thetas)
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
                gradient = self.gradient_(x, y)
                # 4. calculate and assign the new thetas
                self.thetas -= self.alpha * gradient
            return self.thetas
        except:
            return None


# -----------------------------------------------------------------------------
# MyLinearRegression class with l2 regularization : MyRidge
# -----------------------------------------------------------------------------
class MyRidge(MyLinearRegression):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """

    @type_validator
    @shape_validator({'thetas': ('n', 1)})
    def __init__(self, thetas: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000, lambda_: float = 0.5):
        super().__init__(thetas, alpha=alpha, max_iter=max_iter)
        self.lambda_ = lambda_

    def get_params_(self) -> tuple:
        """Returns a tuple with the parameters of the model"""
        return (self.alpha, self.max_iter, self.lambda_, self.thetas)

    @type_validator
    @shape_validator({'thetas': ('n', 1)})
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
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    @regularize_loss
    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return super().loss_(y, y_hat)

    @type_validator
    @shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
    @regularize_grad
    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return super().gradient_(x, y)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
# this module is tested in next exercise

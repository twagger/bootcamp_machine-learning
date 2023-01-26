"""Linear loss regression module"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
# nd arrays
import numpy as np
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex01'))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex06'))
from ridge import type_validator, shape_validator
from l2_reg import l2


# -----------------------------------------------------------------------------
# Regularized linear regression loss
# -----------------------------------------------------------------------------
@type_validator
@shape_validator({'y': ('m', 1), 'y_hat': ('m', 1), 'theta': ('n', 1)})
def reg_loss_(y: np.ndarray, y_hat: np.ndarray, theta: np.ndarray,
              lambda_: float) -> float:
    """
    Computes the regularized loss of a linear regression model from two
            non-empty numpy.array, without any for loop.Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    try:
        m, _ = y.shape
        loss = float((((y_hat - y).T.dot(y_hat - y)) / (2 * m))[0][0])
        regularization_term = (lambda_ / (2 * m)) * l2(theta)
        return loss + regularization_term
    except:
        return None


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    # Example :
    print(f'ex00: {reg_loss_(y, y_hat, theta, .5)}\n')
    # Output:
    # 0.8503571428571429

    # Example :
    print(f'ex01: {reg_loss_(y, y_hat, theta, .05)}\n')
    # Output:
    # 0.5511071428571429

    # Example :
    print(f'ex02: {reg_loss_(y, y_hat, theta, .9)}\n')
    # Output:
    # 1.116357142857143

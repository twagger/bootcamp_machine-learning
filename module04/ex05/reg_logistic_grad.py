"""Regularized logistic gradient"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
# nd arrays
import numpy as np
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'ex06'))
from ridge import type_validator, shape_validator


# -----------------------------------------------------------------------------
# Regularized logistic regression gradient
# -----------------------------------------------------------------------------
@type_validator
@shape_validator({'y': ('m', 1), 'x': ('m', 'n'), 'theta': ('n + 1', 1)})
def reg_logistic_grad(y: np.ndarray, x: np.ndarray, theta: np.ndarray,
                      lambda_: float) -> np.ndarray:
    """
    Computes the regularized logistic gradient of three non-empty
            numpy.ndarray, with two for-loops. The three arrayArgs:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the
         formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    try:
        m, n = x.shape
        x_prime = np.hstack((np.ones((m, 1)), x))
        grad = []
        # calculate the gradient vector by looping on each feature and data
        for j in range(n + 1):
            derv = 0.0
            for i in range(m):
                y_hat_i = 1 / (1 + np.exp(-(x_prime[i, :].dot(theta))))
                derv += (y_hat_i - y[i][0]) * x_prime[i][j]
            regul = lambda_ * theta[j][0] if j > 0 else 0
            grad.append((derv + regul) / m)
        return np.array(grad)

    except (ValueError, TypeError) as exc:
        print(exc)
        return None


@type_validator
@shape_validator({'y': ('m', 1), 'x': ('m', 'n'), 'theta': ('n + 1', 1)})
def vec_reg_logistic_grad(y: np.ndarray, x: np.ndarray, theta: np.ndarray,
                          lambda_: float) -> np.ndarray:
    """
    Computes the regularized logistic gradient of three non-empty
            numpy.ndarray, without any for-loop. The three arrArgs:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of shape m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the
         formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    try:
        m, _ = x.shape
        x_prime = np.hstack((np.ones((m, 1)), x))
        theta_prime = theta.copy()
        theta_prime[0][0] = 0
        y_hat = 1 / (1 + np.exp(-(x_prime.dot(theta))))
        return (x_prime.T.dot(y_hat - y) + lambda_ * theta_prime) / m

    except (ValueError, TypeError) as exc:
        print(exc)
        return None


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    x = np.array([[0, 2, 3, 4],
                [2, 4, 5, 5],
                [1, 3, 2, 7]])

    y = np.array([[0],
                [1],
                [1]])

    theta = np.array([[-2.4],
                    [-1.5],
                    [0.3],
                    [-1.4],
                    [0.7]])
    # Example 1.1:
    print(f'ex1.1:\n{reg_logistic_grad(y, x, theta, 1.0)}\n')
    # Output:
    # array([[-0.55711039],
    #        [-1.40334809],
    #        [-1.91756886],
    #        [-2.56737958],
    #        [-3.03924017]])

    # Example 1.2:
    print(f'ex1.2:\n{vec_reg_logistic_grad(y, x, theta, 1.0)}\n')
    # Output:
    # array([[-0.55711039],
    #        [-1.40334809],
    #        [-1.91756886],
    #        [-2.56737958],
    #        [-3.03924017]])

    # Example 2.1:
    print(f'ex2.1:\n{reg_logistic_grad(y, x, theta, 0.5)}\n')
    # Output:
    # array([[-0.55711039],
    #        [-1.15334809],
    #        [-1.96756886],
    #        [-2.33404624],
    #        [-3.15590684]])

    # Example 2.2:
    print(f'ex2.2:\n{vec_reg_logistic_grad(y, x, theta, 0.5)}\n')
    # Output:
    # array([[-0.55711039],
    #        [-1.15334809],
    #        [-1.96756886],
    #        [-2.33404624],
    #        [-3.15590684]])

    # Example 3.1:
    print(f'ex3.1:\n{reg_logistic_grad(y, x, theta, 0.0)}\n')
    # Output:
    # array([[-0.55711039],
    #        [-0.90334809],
    #        [-2.01756886],
    #        [-2.10071291],
    #        [-3.27257351]])

    # Example 3.2:
    print(f'ex3.2:\n{vec_reg_logistic_grad(y, x, theta, 0.0)}\n')
    # Output:
    # array([[-0.55711039],
    #        [-0.90334809],
    #        [-2.01756886],
    #        [-2.10071291],
    #        [-3.27257351]])

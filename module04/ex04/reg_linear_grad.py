"""Regularized linear gradient"""
import sys
import numpy as np


def input_validator(func):
    """Input validator for np arrays"""
    def wrapper(*args, **kwargs):
        y, x, theta, lambda_ = args
        if (not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray)
                or not isinstance(theta, np.ndarray)):
            print('Something went wrong', file=sys.stderr)
            return None
        if (y.shape[1] != 1 or theta.shape[1] != 1
                or x.shape[0] != y.shape[0]
                or theta.shape[0] != x.shape[1] + 1):
            print('Something went wrong', file=sys.stderr)
            return None
        if not isinstance(lambda_, float):
            print('Something went wrong', file=sys.stderr)
            return None
        return func(*args, **kwargs)
    return wrapper

@input_validator
def reg_linear_grad(y: np.ndarray, x: np.ndarray, theta: np.ndarray,
                    lambda_: float) -> np.ndarray:
    """
    Computes the regularized linear gradient of three non-empty numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results
            of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # computation
        m, _ = x.shape
        x_prime = np.hstack((np.ones((m, 1)), x))
        grad = []
        # calculate the gradient vector by looping on each feature and data
        for j in range(x_prime.shape[1]):
            derv = 0.0
            for i in range(m):
                y_hat_i = x_prime[i, :].dot(theta)
                derv += (y_hat_i - y[i][0]) * x_prime[i][j]
            regul = lambda_ * theta[j][0] if j > 0 else 0
            grad.append((derv + regul) / m)
        return np.array(grad)

    except (ValueError, TypeError) as exc:
        print(exc)
        return None

@input_validator
def vec_reg_linear_grad(y: np.ndarray, x: np.ndarray, theta: np.ndarray,
                        lambda_: float) -> np.ndarray:
    """
    Computes the regularized linear gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results
            of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # computation
        m, _ = x.shape
        x_prime = np.hstack((np.ones((m, 1)), x))
        theta_prime = theta.copy()
        theta_prime[0][0] = 0
        y_hat = x_prime.dot(theta)
        return (x_prime.T.dot(y_hat - y) + lambda_ * theta_prime) / m

    except (ValueError, TypeError) as exc:
        print(exc)
        return None


if __name__ == "__main__":

    x = np.array([[ -6, -7, -9],
                  [ 13, -2, 14],
                  [ -7, 14, -1],
                  [ -8, -4, 6],
                  [ -5, -9, 6],
                  [ 1, -5, 11],
                  [ 9, -11, 8]])

    y = np.array([[2],
                  [14],
                  [-13],
                  [5],
                  [12],
                  [4],
                  [-19]])

    theta = np.array([[7.01],
                      [3],
                      [10.5],
                      [-6]])

    # Example 1.1:
    print(f'ex1.1:\n{reg_linear_grad(y, x, theta, 1.)}\n')
    # Output:
    # array([[ -60.99 ],
    #        [-195.64714286],
    #        [ 863.46571429],
    #        [-644.52142857]])

    # Example 1.2:
    print(f'ex1.2:\n{vec_reg_linear_grad(y, x, theta, 1.0)}\n')
    # Output:
    # array([[ -60.99 ],
    #        [-195.64714286],
    #        [ 863.46571429],
    #        [-644.52142857]])

    # Example 2.1:
    print(f'ex2.1:\n{reg_linear_grad(y, x, theta, 0.5)}\n')
    # Output:
    # array([[ -60.99 ],
    #        [-195.86142857],
    #        [ 862.71571429],
    #        [-644.09285714]])

    # Example 2.2:
    print(f'ex2.2:\n{vec_reg_linear_grad(y, x, theta, 0.5)}\n')
    # Output:
    # array([[ -60.99 ],
    #        [-195.86142857],
    #        [ 862.71571429],
    #        [-644.09285714]])

    # Example 3.1:
    print(f'ex3.1:\n{reg_linear_grad(y, x, theta, 0.0)}\n')
    # Output:
    # array([[ -60.99 ],
    #        [-196.07571429],
    #        [ 861.96571429],
    #        [-643.66428571]])

    # Example 3.2:
    print(f'ex3.2:\n{vec_reg_linear_grad(y, x, theta, 0.0)}\n')
    # Output:
    # array([[ -60.99 ],
    #        [-196.07571429],
    #        [ 861.96571429],
    #        [-643.66428571]])

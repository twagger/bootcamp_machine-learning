"""Regularized linear gradient"""
import numpy as np


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
        # shape test
        if (x.shape[0] != y.shape[0] or y.shape[1] != 1
                or theta.shape[0] != x.shape[1] + 1 or theta.shape[1] != 1):
            print('Error: wrong shape on parameter(s)')
            return None
        # computation
        x_prime = np.hstack((np.ones((y.shape)), x))
        result = []
        # calculate the derivative by looping on each feature and each data
        for j in range(x.shape[1] + 1):
            regul = lambda_ * theta[j][0]
            error_ = 0
            for i in range(x.shape[0]):
                error_ += (x_prime[i, :].dot(theta) - y[i][0]) * x_prime[i][j]
            result.append([((error_ + regul) / x.shape[0])[0]])
        return np.array(result)

    except (ValueError, TypeError) as exc:
        print(exc)
        return None

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
        # shape test
        if (x.shape[0] != y.shape[0] or y.shape[1] != 1
                or theta.shape[0] != x.shape[1] + 1 or theta.shape[1] != 1):
            print('Error: wrong shape on parameter(s)')
            return None
        # computation
        x_prime = np.hstack((np.ones((y.shape)), x))
        theta_prime = theta
        theta_prime[0][0] = 0
        y_hat = x_prime.dot(theta)
        return (x_prime.T.dot(y_hat - y) + lambda_ * theta_prime) / x.shape[0]

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
    print(f'ex1.1:\n{reg_linear_grad(y, x, theta, 1)}\n')
    # Output:
    # array([[ -60.99 ],
    #        [-195.64714286],
    #        [ 863.46571429],
    #        [-644.52142857]])
    
    # Example 1.2:
    print(f'ex1.2:\n{vec_reg_linear_grad(y, x, theta, 1)}\n')
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

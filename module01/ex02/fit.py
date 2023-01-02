"""Fit module"""
import numpy as np


def predict_(x: np.ndarray, theta:np.ndarray) -> np.ndarray:
    """
    Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
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
        if x.shape[1] != 1 or theta.shape[0] != 2 or theta.shape[1] != 1:
            print('Error: wrong shape on parameter(s)')
            return None
        # creation of the prediction matrix
        x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
        return x_prime.dot(theta)

    except (TypeError, ValueError) as exc:
        print(exc)
        return None

def fit_(x: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float,
         max_iter: int):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1:
            (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1:
            (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the
            gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # type test
        if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
                or not isinstance(theta, np.ndarray)
                or not isinstance(alpha, float)
                or not isinstance(max_iter, int)):
            print('Error: wrong type')
            return None
        # shape test
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        theta = theta.reshape(2, 1)
        if x.shape != y.shape or x.shape[1] != 1 or theta.shape != (2, 1):
            print('Error: wrong shape on parameter(s)')
            return None
        # calculation of the gradient vector
        x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
        # loop max_iter times
        for _ in range(max_iter):
            # calculate the grandient for current thetas
            gradient = x_prime.T.dot(x_prime.dot(theta) - y) / x.shape[0]
            # calculate and assign the new thetas
            theta[0][0] -= alpha * gradient[0][0]
            theta[1][0] -= alpha * gradient[1][0]
        return theta

    except (TypeError, ValueError):
        print('Error: wrong shape on parameter(s)')
        return None


if __name__ == "__main__":

    x = np.array([[12.4956442],
                  [21.5007972],
                  [31.5527382],
                  [48.9145838],
                  [57.5088733]])

    y = np.array([[37.4013816],
                  [36.1473236],
                  [45.7655287],
                  [46.6793434],
                  [59.5585554]])

    theta = np.array([1.0, 1.0]).reshape((-1, 1))

    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    np.testing.assert_allclose(theta1, np.array([[1.40709365],
                                                 [1.1150909 ]]), rtol=1e-8)

    # Example 1:
    np.testing.assert_allclose(predict_(x, theta1),
                               np.array([[15.3408728 ],
                                         [25.38243697],
                                         [36.59126492],
                                         [55.95130097],
                                         [65.53471499]]), rtol=1e-9)

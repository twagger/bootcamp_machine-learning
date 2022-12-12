"""Prediction module"""
import numpy as np


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    # emptyness and type check
    if (not isinstance(x, np.ndarray)
            or (isinstance(x, np.ndarray) and len(x) == 0)):
        print(f'Error: x has to be a numpy.array of dimension m * n')
        return None
    # computation
    ones_column = np.full((x.shape[0], 1), 1.)
    return np.hstack((ones_column,
                      x.reshape(x.shape[0], 1 if x.ndim == 1 else x.shape[1])))

def predict_(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
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
    # type test
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        print('Error: wrong type on parameter(s)')
        return None
    # emptyness test
    if len(x) == 0 or len(theta) == 0:
        print('Error: empty parameter(s)')
        return None
    # shape test
    try:
        x = x.reshape(x.shape[0], 1)
        theta = theta.reshape(2, 1)
    except:
        print('Error: wrong shape on parameter(s)')
        return None
    # creation of the prediction matrix
    return add_intercept(x).dot(theta)


if __name__ == "__main__":

    x = np.arange(1,6)

    # Example 1:
    theta1 = np.array([[5], [0]])
    assert(predict_(x, theta1).all() == np.array([[5.],
                                                  [5.],
                                                  [5.],
                                                  [5.],
                                                  [5.]]).all())

    # Example 2:
    theta2 = np.array([[0], [1]])
    assert(predict_(x, theta2).all() == np.array([[1.],
                                                  [2.],
                                                  [3.],
                                                  [4.],
                                                  [5.]]).all())

    # Example 3:
    theta3 = np.array([[5], [3]])
    assert(predict_(x, theta3).all() == np.array([[ 8.],
                                                  [11.],
                                                  [14.],
                                                  [17.],
                                                  [20.]]).all())

    # Example 4:
    theta4 = np.array([[-3], [1]])
    assert(predict_(x, theta4).all() == np.array([[-2.],
                                                  [-1.],
                                                  [ 0.],
                                                  [ 1.],
                                                  [ 2.]]).all())

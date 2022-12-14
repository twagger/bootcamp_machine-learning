"""Plot module"""
import numpy as np


def simple_gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes a gradient vector from three non-empty numpy.array, with a
    for-loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    # type test
    if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
            or not isinstance(theta, np.ndarray)):
        print('Error: wrong type on parameter(s)')
        return None
    # emptyness test
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        print('Error: empty parameter(s)')
        return None
    # shape test
    try:
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        theta = theta.reshape(2, 1)
    except ValueError:
        print('Error: wrong shape on parameter(s)')
        return None
    if x.shape[0] != y.shape[0]:
        print('Error: wrong shape on parameter(s)')
        return None
    # calculation of the gradient vector
    ones_column = np.ones((x.shape[0], 1))
    x_matrix = np.hstack((ones_column, x))
    grad_0 = np.sum(x_matrix.dot(theta) - y) / x.shape[0]
    grad_1 = np.sum((x_matrix.dot(theta) - y) * x) / x.shape[0]
    return np.array([grad_0, grad_1]).reshape((2, 1))


if __name__ == "__main__":

    x = np.array([12.4956442,
                  21.5007972,
                  31.5527382,
                  48.9145838,
                  57.5088733]).reshape((-1, 1))

    y = np.array([37.4013816,
                  36.1473236,
                  45.7655287,
                  46.6793434,
                  59.5585554]).reshape((-1, 1))

    # Example 0:
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(simple_gradient(x, y, theta1))
    # Output:
    # array([[-19.0342574], [-586.66875564]])

    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(simple_gradient(x, y, theta2))
    # Output:
    # array([[-57.86823748], [-2230.12297889]])

"""Plot module"""
import numpy as np


def simple_gradient(x: np.ndarray, y: np.ndarray,
                    theta: np.ndarray) -> np.ndarray:
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
    Raises:
        This function should not raise any Exception.
    """
    try:
        # type test
        if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
                or not isinstance(theta, np.ndarray)):
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
        return x_prime.T.dot(x_prime.dot(theta) - y) / x.shape[0]

    except (TypeError, ValueError, AttributeError):
        print('Error: wrong shape on parameter(s)')
        return None

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
    np.testing.assert_allclose(simple_gradient(x, y, theta1),
                               np.array([[-19.0342574],
                                         [-586.66875564]]), rtol=1e-9)

    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    np.testing.assert_allclose(simple_gradient(x, y, theta2),
                               np.array([[-57.86823748],
                                         [-2230.12297889]]), rtol=1e-9)

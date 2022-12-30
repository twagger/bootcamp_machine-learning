"""Intercept module"""
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
    ones_column = np.ones((x.shape[0], 1))
    return np.hstack((ones_column,
                      x.reshape(x.shape[0], 1 if x.ndim == 1 else x.shape[1])))


if __name__ == "__main__":

    # Test 1
    x = np.arange(1,6).reshape((-1, 1))
    assert(add_intercept(x).all() == np.array([[1., 1.],
                                               [1., 2.],
                                               [1., 3.],
                                               [1., 4.],
                                               [1., 5.]]).all())

    # Test 2
    y = np.arange(1,10).reshape((3,3))
    assert(add_intercept(y).all() == np.array([[1., 1., 2., 3.],
                                               [1., 4., 5., 6.],
                                               [1., 7., 8., 9.]]).all())

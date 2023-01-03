"""Loss module"""
import numpy as np


def loss_(y: np.ndarray, y_hat: np.ndarray) -> float:
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
        # shapes
        y = y.reshape((-1, 1))
        y_hat = y_hat.reshape((-1, 1))
        # calculation
        return float((((y_hat - y).T.dot(y_hat - y))
                        / (2 * y.shape[0]))[0][0])
    except (ValueError, TypeError, AttributeError):
        return None


if __name__ == "__main__":

    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    # Example 1:
    assert loss_(X, Y) == 2.142857142857143

    # Example 2:
    assert loss_(X, X) == 0.0
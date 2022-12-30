"""Prediction module"""
import numpy as np


def predict_(x, theta):
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

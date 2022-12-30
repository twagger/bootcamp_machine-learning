"""Prediction module"""
import numpy as np


def simple_predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
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
    try:
        # shape test
        if x.shape[1] != 1 or theta.shape[0] != 2 or theta.shape[1] != 1:
            print('Error: wrong shape on parameter(s)')
            return None
        # creation of the prediction matrix using a vectorized lambda function
        predict_formula = lambda x, t1, t2: t1 + t2 * x
        vfunc = np.vectorize(predict_formula)
        return vfunc(x, theta[0], theta[1])
    except (TypeError, ValueError) as exc:
        print(exc)
        return None


if __name__ == "__main__":

    x = np.arange(1, 6).reshape((-1, 1))

    # Example 1:
    theta1 = np.array([[5], [0]]).reshape((-1, 1))
    np.testing.assert_array_equal(simple_predict(x, theta1), np.array([[5.],
                                                                       [5.],
                                                                       [5.],
                                                                       [5.],
                                                                       [5.]]))

    # Example 2:
    theta2 = np.array([[0], [1]]).reshape((-1, 1))
    np.testing.assert_array_equal(simple_predict(x, theta2), np.array([[1.],
                                                                       [2.],
                                                                       [3.],
                                                                       [4.],
                                                                       [5.]]))

    # Example 3:
    theta3 = np.array([[5], [3]]).reshape((-1, 1))
    np.testing.assert_array_equal(simple_predict(x, theta3), np.array([[ 8.],
                                                                       [11.],
                                                                       [14.],
                                                                       [17.],
                                                                       [20.]]))

    # Example 4:
    theta4 = np.array([[-3], [1]]).reshape((-1, 1))
    np.testing.assert_array_equal(simple_predict(x, theta4), np.array([[-2.],
                                                                       [-1.],
                                                                       [ 0.],
                                                                       [ 1.],
                                                                       [ 2.]]))

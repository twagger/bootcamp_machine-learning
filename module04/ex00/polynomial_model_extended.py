"""Polynomial model feature"""
import sys
import numpy as np


def add_polynomial_features(x: np.ndarray, power: int) -> np.ndarray:
    """
    Add polynomial features to matrix x by raising its columns to every power
            in the range of 1 up to the power giveArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        power: has to be an int, the power up to which the columns of matrix x
            are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray,
            of shape m * (np), containg the polynomial feature vaNone if x is
            an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # type check
        if not isinstance(x, np.ndarray) or not isinstance(power, int):
            print("Something went wrong", file=sys.stderr)
            return None
        # calculation
        result = x.copy()
        for i in range(power - 1):
            result = np.c_[result, x ** (2 + i)]
        return result

    except (ValueError, TypeError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None


if __name__ == "__main__":

    x = np.arange(1,11).reshape(5, 2)

    # Example 1:
    print(f'ex01:\n{add_polynomial_features(x, 3)}\n')
    # Output:
    # array([[ 1, 2, 1, 4, 1, 8],
    #        [ 3, 4, 9, 16, 27, 64],
    #        [ 5, 6, 25, 36, 125, 216],
    #        [ 7, 8, 49, 64, 343, 512],
    #        [ 9, 10, 81, 100, 729, 1000]])

    # Example 2:
    print(f'ex02:\n{add_polynomial_features(x, 4)}\n')
    # Output:
    # array([[ 1, 2, 1, 4, 1, 8, 1, 16],
    #        [ 3, 4, 9, 16, 27, 64, 81, 256],
    #        [ 5, 6, 25, 36, 125, 216, 625, 1296],
    #        [ 7, 8, 49, 64, 343, 512, 2401, 4096],
    #        [ 9, 10, 81, 100, 729, 1000, 6561, 10000]])

"""Sigmoid module"""
import numpy as np
import math


def sigmoid_(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # shape test
        if x.shape[1] != 1:
            print('Error: wrong shape on parameter(s)')
            return None
        # calculation
        return (1 / (1 + math.e ** -x))
    except (TypeError, ValueError) as exc:
        print(exc)
        return None


if __name__ == "__main__":

    # Example 1:
    x = np.array([[-4]])
    print(f'ex01:\n{sigmoid_(x)}\n')
    # Output:
    # array([[0.01798620996209156]])

    # Example 2:
    x = np.array([[2]])
    print(f'ex02:\n{sigmoid_(x)}\n')
    # Output:
    # array([[0.8807970779778823]])

    # Example 3:
    x = np.array([[-4], [2], [0]])
    print(f'ex03:\n{sigmoid_(x)}\n')
    # Output:
    # array([[0.01798620996209156], [0.8807970779778823], [0.5]])

"""L2 regression module"""
import numpy as np


def iterative_l2(theta: np.ndarray) -> float:
    """
    Computes the L2 regularization of a non-empty numpy.ndarray, with a
        for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # shape test
        if theta.shape[1] != 1:
            print('Error: wrong shape on parameter(s)')
            return None
        # iterative l2
        result = 0
        for i in range(theta.shape[0]):
            if i != 0:
                result += theta[i][0] ** 2
        return result

    except (ValueError, TypeError) as exc:
        print(exc)
        return None

def l2(theta: np.ndarray) -> float:
    """
    Computes the L2 regularization of a non-empty numpy.ndarray, without any
        for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # shape test
        if theta.shape[1] != 1:
            print('Error: wrong shape on parameter(s)')
            return None
        # l2
        theta_prime = theta
        theta_prime[0][0] = 0
        return theta_prime.T.dot(theta_prime)[0][0]

    except (ValueError, TypeError) as exc:
        print(exc)
        return None

if __name__ == "__main__":

    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    # Example 1:
    print(f'ex01: {iterative_l2(x)}\n')
    # Output:
    # 911.0

    # Example 2:
    print(f'ex02: {l2(x)}\n')
    # Output:
    # 911.0

    y = np.array([3,0.5,-6]).reshape((-1, 1))
    # Example 3:
    print(f'ex03: {iterative_l2(y)}\n')
    # Output:
    # 36.25

    # Example 4:
    print(f'ex04: {l2(y)}\n')
    # Output:
    # 36.25

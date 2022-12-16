"""Linear loss regression mmodule"""
import numpy as np
import sys
sys.path.insert(1, '../ex01/')
from l2_reg import l2


def reg_log_loss_(y: np.ndarray, y_hat: np.ndarray, theta: np.ndarray,
                  lambda_: float) -> float:
    """
    Computes the regularized loss of a logistic regression model from two
            non-empty numpy.ndarray, without any for lArgs:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # shape test
        if (y.shape != y_hat.shape or theta.shape[1] != 1):
            print('Error: wrong shape on parameter(s)')
            return None
        # compputation
        eps: float=1e-15
        non_reg_loss = float((-(1 / y.shape[0])
                              * (y.T.dot(np.log(y_hat + eps))
                              + (1 - y).T.dot(np.log(1 - y_hat + eps))))[0][0])
        reg_loss = (non_reg_loss + (lambda_ / (2 * y.shape[0])) * l2(theta))
        return reg_loss

    except (ValueError, TypeError) as exc:
        print(exc)
        return None


if __name__ == "__main__":

    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    # Example :
    print(f'ex00: {reg_log_loss_(y, y_hat, theta, .5)}\n')
    # Output:
    # 0.43377043716475955

    # Example :
    print(f'ex01: {reg_log_loss_(y, y_hat, theta, .05)}\n')
    # Output:
    # 0.13452043716475953

    # Example :
    print(f'ex02: {reg_log_loss_(y, y_hat, theta, .9)}\n')
    # Output:
    # 0.6997704371647596

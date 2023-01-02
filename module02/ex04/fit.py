"""Fit module"""
import numpy as np


def fit_(x: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float,
         max_iter: int):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
            (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
            (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
            (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the
            gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension (nb of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This  function should not raise any Exception.
    """
    try:
        # shape test
        if (x.shape[0] != y.shape[0] or y.shape[1] != 1
                or theta.shape[0] != x.shape[1] + 1
                or theta.shape[1] != 1):
            print('Error: wrong shape on parameter(s)')
            return None
        # calculation of the gradient vecto
        # 1. X to X'
        x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
        # 2. loop
        for _ in range(max_iter):
            # 3. calculate the grandient for current thetas
            gradient = x_prime.T.dot(x_prime.dot(theta) - y) / x.shape[0]
            # 4. calculate and assign the new thetas
            theta -= alpha * gradient
        return theta
    except (ValueError, TypeError) as exc:
        print(exc)
        return None


if __name__ == "__main__":

    import sys
    sys.path.insert(1, '../ex00/')
    from prediction import predict_

    x = np.array([[0.2, 2., 20.],
                  [0.4, 4., 40.],
                  [0.6, 6., 60.],
                  [0.8, 8., 80.]])

    y = np.array([[19.6],
                  [-2.8],
                  [-25.2],
                  [-47.6]])

    theta = np.array([[42.], [1.], [1.], [1.]])

    # Example 0:
    theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
    print(theta2)
    # Output:
    # array([[41.99..],
    #        [0.97...],
    #        [0.77...],
    #        [-1.20..]])

    # Example 1:
    print(predict_(x, theta2))
    # Output:
    # array([[19.5992...],
    #        [-2.8003...],
    #        [-25.1999..],
    #        [-47.5996..]])

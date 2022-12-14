"""Fit module"""
import numpy as np


def fit_(x: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float,
         max_iter: int):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1:
            (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1:
            (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the
            gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    # type test
    if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
            or not isinstance(theta, np.ndarray)
            or not isinstance(alpha, float)
            or not isinstance(max_iter, int)):
        print('Error: wrong type on parameter(s)')
        return None
    # emptyness test
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        print('Error: empty parameter(s)')
        return None
    # shape test
    try:
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        theta = theta.reshape(2, 1)
    except ValueError:
        print('Error: wrong shape on parameter(s)')
        return None
    if x.shape[0] != y.shape[0]:
        print('Error: wrong shape on parameter(s)')
        return None
    # calculation of the gradient vecto
    # 1. x to matrix
    ones_column = np.ones((x.shape[0], 1))
    x_matrix = np.hstack((ones_column, x))
    # 2. loop
    for _ in range(max_iter):
        # 3. calculate the grandient for current thetas
        gradient = x_matrix.T.dot(x_matrix.dot(theta) - y) / x.shape[0]
        # 4. calculate and assign the new thetas
        theta[0][0] -= alpha * gradient[0][0]
        theta[1][0] -= alpha * gradient[1][0]
    return theta


if __name__ == "__main__":

    import sys
    sys.path.insert(1, '../../module00/ex02/')
    from prediction import predict_

    x = np.array([[12.4956442],
                  [21.5007972],
                  [31.5527382],
                  [48.9145838],
                  [57.5088733]])

    y = np.array([[37.4013816],
                  [36.1473236],
                  [45.7655287],
                  [46.6793434],
                  [59.5585554]])

    theta = np.array([1.0, 1.0]).reshape((-1, 1))

    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1)
    # Output:
    # array([[1.40709365],
    #        [1.1150909 ]])

    # Example 1:
    print(predict_(x, theta1))
    # Output:
    # array([[15.3408728 ],
    #        [25.38243697],
    #        [36.59126492],
    #        [55.95130097],
    #        [65.53471499]])

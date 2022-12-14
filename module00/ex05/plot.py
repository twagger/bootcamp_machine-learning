"""Plot module"""
import numpy as np
import matplotlib.pyplot as plt


def plot(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """
    Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exceptions.
    """
    # type test
    if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
            or not isinstance(theta, np.ndarray)):
        print('Error: wrong type on parameter(s)')
        return None
    # emptyness test
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        print('Error: empty parameter(s)')
        return None
    # shape test
    try:
        x = x.reshape(x.shape[0], 1)
        y = y.reshape(y.shape[0], 1)
        theta = theta.reshape(2, 1)
    except ValueError:
        print('Error: wrong shape on parameter(s)')
        return None
    # make prediction
    ones_column = np.ones((x.shape[0], 1))
    data = np.hstack((ones_column, x))
    prediction = data.dot(theta)

    # plot
    plt.figure()
    plt.scatter(x, y, marker='o')
    plt.plot(x, prediction, color='red')
    plt.show()


if __name__ == "__main__":

    x = np.arange(1,6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])

    # Example 1:
    theta1 = np.array([[4.5],[-0.2]])
    plot(x, y, theta1)

    # Example 2:
    theta2 = np.array([[-1.5],[2]])
    plot(x, y, theta2)

    # Example 3:
    theta3 = np.array([[3],[0.3]])
    plot(x, y, theta3)
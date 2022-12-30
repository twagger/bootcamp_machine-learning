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
    try:
        # shape test
        if (x.shape[0] != y.shape[0]
                or x.shape[1] != y.shape[1] != theta.shape[1] != 1
                or theta.shape[0] != 2):
            print('Error: wrong shape on parameter(s)')
            return None
        # creation of the prediction matrix
        x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
        prediction = x_prime.dot(theta)
        # plot
        plt.figure()
        plt.scatter(x, y, marker='o')
        plt.plot(x, prediction, color='red')
        plt.show()

    except (TypeError, ValueError) as exc:
        print(exc)
        return None
   

if __name__ == "__main__":

    x = np.arange(1, 6).reshape((-1, 1))
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434,
                  5.95585554]).reshape((-1, 1))

    # Example 1:
    theta1 = np.array([[4.5],[-0.2]]).reshape((-1, 1))
    plot(x, y, theta1)

    # Example 2:
    theta2 = np.array([[-1.5],[2]]).reshape((-1, 1))
    plot(x, y, theta2)

    # Example 3:
    theta3 = np.array([[3],[0.3]]).reshape((-1, 1))
    plot(x, y, theta3)

"""Plot with prediction and loss"""
import numpy as np
import matplotlib.pyplot as plt


def plot_with_loss(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """
    Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # shape test
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        if (x.shape[0] != y.shape[0]
                or x.shape[1] != y.shape[1] != theta.shape[1] != 1
                or theta.shape[0] != 2):
            print('Error: wrong shape on parameter(s)')
            return None
        # creation of the prediction vector
        x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
        prediction = x_prime.dot(theta)
        # plot
        plt.figure()
        plt.scatter(x, y, marker='o')
        plt.plot(x, prediction, color='red')
        for i in range(len(x)):
            plt.plot((x[i], x[i]), (y[i], prediction[i]), c='red', ls='--')
        plt.show()

    except (TypeError, ValueError) as exc:
        print(exc)
        return None


if __name__ == "__main__":

    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

    # Example 1:
    theta1= np.array([18, -1])
    plot_with_loss(x, y, theta1)

    # Example 2:
    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)

    # Example 3:
    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)

"""My Logistic Regression module"""
import math
import numpy as np
import pandas as pd


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        try:
            self.theta = np.array(theta).reshape((-1, 1))
        except (ValueError, TypeError) as exc:
            print(exc)

    def predict_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the vector of prediction y_hat from two non-empty
        numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a vector of dimension m * n.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # shape test
            if (self.theta.shape[0] != x.shape[1] + 1
                    or self.theta.shape[1] != 1):
                print('Error: wrong shape on parameter(s)')
                return None
            # calculation
            x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
            return 1 / (1 + math.e ** -(x_prime.dot(self.theta)))
        except (TypeError, ValueError) as exc:
            print(exc)
            return None

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Description:
        Calculates the loss by element.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_elem: numpy.array, a vector of dimension (number of the training
                examples,1).
            None if there is a dimension matching problem between X, Y or
                theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
           # shape test
            if (y.shape[0] != y_hat.shape[0] or y.shape[1] != 1
                    or y_hat.shape[1] != 1):
                print('Error: wrong shape on parameter(s)')
                return None
            # add a little value to y_hat to avoid log(0) problem
            eps: float=1e-15
            return -(y * np.log(y_hat + eps)
                     + (1 - y) * np.log(1 - y_hat + eps))
        except (ValueError, TypeError):
            return None

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Compute the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
            The logistic loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # shape test
            if (y.shape[0] != y_hat.shape[0] or y.shape[1] != 1
                    or y_hat.shape[1] != 1):
                print('Error: wrong shape on parameter(s)')
                return None
            # add a little value to y_hat to avoid log(0) problem
            eps: float=1e-15
            return float((-(1 / y.shape[0])
                        * (y.T.dot(np.log(y_hat + eps))
                            + (1 - y).T.dot(np.log(1 - y_hat + eps))))[0][0])
        except (TypeError, ValueError) as exc:
            print(exc)
            return None

    def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * n:
                (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
        Returns:
            new_theta: numpy.ndarray, a vector of dimension (n + 1) * 1.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # shape test
            if (x.shape[0] != y.shape[0] or y.shape[1] != 1):
                print('Error: wrong shape on parameter(s)')
                return None
            # calculation of the gradient vector
            # 1. X to X'
            m = x.shape[0]
            x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
            # 2. loop
            for _ in range(self.max_iter):
                # 3. calculate the grandient for current thetas
                y_hat = self.predict_(x)
                gradient = x_prime.T.dot(y_hat - y) / m
                # 4. calculate and assign the new thetas
                self.theta -= self.alpha * gradient
            return self.theta
        except (ValueError, TypeError) as exc:
            print(exc)
            return None


if __name__ == "__main__":

    # setting parameters
    X = np.array([[1., 1., 2., 3.],
                  [5., 8., 13., 21.],
                  [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLogisticRegression(thetas)

    print(f'dataset Y:\n{Y}\n')

    # Example 0:
    y_hat = mylr.predict_(X)
    print(f'firt predict:\n{y_hat}\n')
    # Output:
    # array([[0.99930437],
    #        [1. ],
    #        [1. ]])

    # Example 1:
    print(f'loss:\n{mylr.loss_(Y, y_hat)}\n')
    print(f'loss by elem:\n{mylr.loss_elem_(Y, y_hat)}\n')
    # Output:
    # 11.513157421577004

    # Example 2:
    mylr.fit_(X, Y)
    print(f'fitted thetas:\n{mylr.theta}\n')
    # Output:
    # array([[ 2.11826435]
    #        [ 0.10154334]
    #        [ 6.43942899]
    #        [-5.10817488]
    #        [ 0.6212541 ]])

    # Example 3:
    y_hat = mylr.predict_(X)
    print(f'new prediction:\n{y_hat}\n')
    # Output:
    # array([[0.57606717]
    #        [0.68599807]
    #        [0.06562156]])

    # Example 4:
    print(f'new loss:\n{mylr.loss_(Y, y_hat)}\n')
    # Output:
    # 1.4779126923052268

    # ------------------------
    # Test on a bigger dataset
    # ------------------------
    df = pd.read_csv('./andrew_ng_dataset.csv')

    # setting parameters
    X = np.array(df[['col1','col2']])
    Y = np.array(df[['res']])
    # thetas = np.array([[-25.16133334], [0.20623171], [0.2014716]])
    thetas = np.array([[-6.74692366], [0.24194933], [0.12629922]])
    mylr = MyLogisticRegression(thetas, alpha=4e-3 ,max_iter=10000)

    # Example 0:
    y_hat = mylr.predict_(X)
    print(f'firt predict:\n{y_hat}\n')

    # Example 1:
    first_loss = mylr.loss_(Y, y_hat)
    print(f'loss:\n{first_loss}\n')

    # Example 2:
    mylr.fit_(X, Y)
    print(f'fitted thetas:\n{mylr.theta}\n')

    # Example 3:
    y_hat = mylr.predict_(X)
    print(f'new prediction:\n{y_hat}\n')

    # Example 4:
    new_loss = mylr.loss_(Y, y_hat)
    print(f'new loss:\n{new_loss}\n')
    print(f'{f"-{first_loss - new_loss}" if new_loss < first_loss else f"+{new_loss - first_loss}" }\n')

    # Do a precision percentage
    y_hat = np.array(list(map(lambda x: 0 if x < 0.5 else 1, y_hat)))
    stat = Y == y_hat
    print(stat.sum() / len(stat))

    # plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(np.array(df['col1']),  Y)
    plt.scatter(np.array(df['col1']),  y_hat)
    plt.subplot(1,2,2)
    plt.scatter(np.array(df['col2']),  Y)
    plt.scatter(np.array(df['col2']),  y_hat)
    plt.show()

"""My Linear Regression Module"""
import numpy as np


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000):
        """Constructor"""
        self.alpha = alpha
        self.max_iter = max_iter
        try:
            self.thetas = np.array(thetas).reshape((-1, 1))
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc)

    def predict_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * n.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x and/or theta are not numpy.array.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            m = x.shape[0]
            x_prime = np.c_[np.ones((m, 1)), x]
            return x_prime.dot(self.thetas)
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc)
            return None

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
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
            y = y.reshape((-1, 1))
            y_hat = y_hat.reshape((-1, 1))
            return (y_hat - y) ** 2
        except (ValueError, TypeError, AttributeError):
            return None

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Computes the half mean squared error of two non-empty numpy.array,
        without any for loop.
        The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            The half mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.array.
            None if y and y_hat does not share the same dimensions.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            y = y.reshape((-1, 1))
            y_hat = y_hat.reshape((-1, 1))
            m = y.shape[0]
            return float((((y_hat - y).T.dot(y_hat - y)) / (2 * m))[0][0])
        except (ValueError, TypeError, AttributeError):
            return None

    def mse_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Computes the mean squared error of two non-empty numpy.array,
        without any for loop.
        The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            The half mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.array.
            None if y and y_hat does not share the same dimensions.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            # shapes
            y = y.reshape((-1, 1))
            y_hat = y_hat.reshape((-1, 1))
            # calculation
            m = y.shape[0]
            return float((((y_hat - y).T.dot(y_hat - y)) / (m))[0][0])
        except (ValueError, TypeError, AttributeError):
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
            x_prime = np.c_[np.ones((m, 1)), x]
            # 2. loop
            for _ in range(self.max_iter):
                # 3. calculate the grandient for current thetas
                gradient = (x_prime.T.dot(x_prime.dot(self.thetas) - y) / m)
                # 4. calculate and assign the new thetas
                self.thetas -= self.alpha * gradient
            return self.thetas
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc)
            return None


if __name__ == "__main__":

    X = np.array([[1., 1., 2., 3.],
                  [5., 8., 13., 21.],
                  [34., 55., 89., 144.]])

    Y = np.array([[23.],
                  [48.],
                  [218.]])

    mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])

    # Example 0:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    # array([[8.],
    #        [48.],
    #        [323.]])

    # Example 1:
    print(mylr.loss_elem_(Y, y_hat))
    # Output:
    # array([[225.],
    #        [0.],
    #        [11025.]])

    # Example 2:
    print(mylr.loss_(Y, y_hat))
    # Output:
    # 1875.0

    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    print(mylr.thetas)
    # Output:
    # array([[18.188..],
    #        [2.767..],
    #        [-0.374..],
    #        [1.392..],
    #        [0.017..]])

    # Example 4:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    # array([[23.417..],
    #        [47.489..],
    #        [218.065...]])

    # Example 5:
    print(mylr.loss_elem_(Y, y_hat))
    # Output:
    # array([[0.174..],
    #        [0.260..],
    #        [0.004..]])

    # Example 6:
    print(mylr.loss_(Y, y_hat))
    # Output:
    # 0.0732..

"""My linear regression module"""
import numpy as np
import matplotlib.pyplot as plt


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
        self.thetas = thetas

    def predict_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x and/or theta are not numpy.array.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            # shapes
            x = x.reshape((-1, 1))
            # 0. add intercept
            ones_column = np.ones((x.shape[0], 1))
            x_matrix = np.hstack((ones_column, x))
            # 1. dot product with thetas
            return x_matrix.dot(self.thetas)
        except (ValueError, TypeError) as exc:
            print(exc)
            return None

    def loss_elem_(self, y, y_hat):
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
            # shapes
            y = y.reshape((-1, 1))
            y_hat = y_hat.reshape((-1, 1))
            # calculation
            return (y_hat - y) ** 2
        except (ValueError, TypeError):
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
            # shapes
            y = y.reshape((-1, 1))
            y_hat = y_hat.reshape((-1, 1))
            # calculation
            return float((((y_hat - y).T.dot(y_hat - y))
                         / (2 * y.shape[0]))[0][0])
        except (ValueError, TypeError):
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
            return float((((y_hat - y).T.dot(y_hat - y))
                         / (y.shape[0]))[0][0])
        except (ValueError, TypeError):
            return None

    def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
        Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # shapes
            x = x.reshape((-1, 1))
            y = y.reshape((-1, 1))
            # 0. add intercept
            m = x.shape[0]
            x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
            # 1. loop
            for _ in range(self.max_iter):
                # 2. calculate the gradient for current thetas
                y_hat = self.predict_(x)
                gradient = x_prime.T.dot(y_hat - y) / m
                # 3. calculate and assign the new thetas
                self.thetas[0][0] -= self.alpha * gradient[0][0]
                self.thetas[1][0] -= self.alpha * gradient[1][0]
            return self.thetas
        except (ValueError, TypeError):
            return None

if __name__ == '__main__':

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

    lr1 = MyLinearRegression(np.array([[2], [0.7]]))

    # Example 0.0:
    y_hat = lr1.predict_(x)
    print(f'{y_hat}\n')
    # Output:
    # array([[10.74695094],
    #        [17.05055804],
    #        [24.08691674],
    #        [36.24020866],
    #        [42.25621131]])

    # Example 0.1:
    print(f'{lr1.loss_elem_(y, y_hat)}\n')
    # Output:
    # array([[710.45867381],
    #        [364.68645485],
    #        [469.96221651],
    #        [108.97553412],
    #        [299.37111101]])

    # Example 0.2:
    print(f'{lr1.loss_(y, y_hat)}\n')
    # Output:
    # 195.34539903032385

    # Example 1.0:
    lr2 = MyLinearRegression(np.array([[1.0], [1.0]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    print(f'{lr2.thetas}\n')
    # Output:
    # array([[1.40709365],
    #        [1.1150909 ]])

    # Example 1.1:
    y_hat = lr2.predict_(x)
    print(f'{y_hat}\n')
    # Output:
    # array([[15.3408728 ],
    #        [25.38243697],
    #        [36.59126492],
    #        [55.95130097],
    #        [65.53471499]])

    # Example 1.2:
    print(f'{lr2.loss_elem_(y, y_hat)}\n')
    # Output:
    # array([[486.66604863],
    #        [115.88278416],
    #        [ 84.16711596],
    #        [ 85.96919719],
    #        [ 35.71448348]])

    # Example 1.3:
    print(f'{lr2.loss_(y, y_hat)}')
    # Output:
    # 80.83996294128525

"""
Ridge regression :  linear regression regularized with L2
"""
import sys
import numpy as np

## Revoir l'input validator et tester les differentes fonctions

# decorator for input validation
def input_validator(func):
    """Input validator for np arrays"""
    def wrapper(*args, **kwargs):
        y, x, theta, lambda_ = args
        if (not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray)
                or not isinstance(theta, np.ndarray)):
            print('Something went wrong', file=sys.stderr)
            return None
        if (y.shape[1] != 1 or theta.shape[1] != 1
                or x.shape[0] != y.shape[0]
                or theta.shape[0] != x.shape[1] + 1):
            print('Something went wrong', file=sys.stderr)
            return None
        if not isinstance(lambda_, float):
            print('Something went wrong', file=sys.stderr)
            return None
        return func(*args, **kwargs)
    return wrapper


# L2 regularization
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
        # type test
        if not isinstance(theta, np.ndarray):
            print('Something went wrong', file=sys.stderr)
            return None
        # shape test
        if theta.shape[1] != 1:
            print('Something went wrong', file=sys.stderr)
            return None
        # l2
        theta_prime = theta
        theta_prime[0][0] = 0
        return theta_prime.T.dot(theta_prime)[0][0]

    except (ValueError, TypeError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None

# my Linear regression class from previous module for inheritance
# normaly I should import this from another file but the files we can return in
# this exercise are limited to this one only so I copied the code directly in 
# it
class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000):
        """Constructor"""
        # type test
        if not isinstance(alpha, float) or not isinstance(max_iter, int):
            print('Something went wrong', file=sys.stderr)
        else:
            self.alpha = alpha
            self.max_iter = max_iter
            try:
                self.thetas = np.array(thetas).reshape((-1, 1))
            except (ValueError, TypeError, AttributeError) as exc:
                print(exc, file=sys.stderr)

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
            # type test
            if not isinstance(x, np.ndarray):
                print('Something went wrong', file=sys.stderr)
                return None
            # shape test
            m, n = x.shape
            if self.thetas.shape[0] != n + 1:
                print('Something went wrong', file=sys.stderr)
                return None
            x_prime = np.c_[np.ones((m, 1)), x]
            return x_prime.dot(self.thetas)
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)
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
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)
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
            m, _ = y.shape
            return float((((y_hat - y).T.dot(y_hat - y)) / (2 * m))[0][0])
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)
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
            m, _ = x.shape
            if (y.shape[0] != m or y.shape[1] != 1):
                print('Error: wrong shape on parameter(s)', file=sys.stderr)
                return None
            # calculation of the gradient vector
            # 1. X to X'
            x_prime = np.c_[np.ones((m, 1)), x]
            # 2. loop
            for _ in range(self.max_iter):
                # 3. calculate the grandient for current thetas
                gradient = (x_prime.T.dot(x_prime.dot(self.thetas) - y) / m)
                # 4. calculate and assign the new thetas
                self.thetas -= self.alpha * gradient
            return self.thetas
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)
            return None


class MyRidge(MyLinearRegression):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """

    def __init__(self, thetas: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000, lambda_: float = 0.5):
        if (not isinstance(alpha, float) or not isinstance(max_iter, int)
                or not isinstance(lambda_, float)
                or not isinstance(thetas, np.ndarray)):
            print('Something went wrong', file=sys.stderr)
            return
        self.alpha = alpha
        self.max_iter = max_iter
        self.lambda_ = lambda_
        try:
            self.thetas = np.array(thetas).reshape((-1, 1))
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)

    def get_params_(self) -> tuple:
        """Returns a tuple with the parameters of the model"""
        return (self.alpha, self.max_iter, self.lambda_, self.thetas)

    def set_params_(self, alpha: float = None, max_iter: int = None,
                    lambda_: float = None, thetas: np.ndarray = None):
        """Update the parameters of the model"""
        try:
            if alpha is not None and isinstance(alpha, float):
                self.alpha = alpha
            if max_iter is not None and isinstance(max_iter, int):
                self.max_iter = max_iter
            if lambda_ is not None and isinstance(lambda_, float):
                self.lambda_ = lambda_
            if thetas is not None and isinstance(thetas, np.ndarray):
                self.thetas = np.array(thetas).reshape((-1, 1))
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)

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
            loss_elem = (y_hat - y) ** 2
            regularization_term = (self.lambda_ / (2 * m)) * l2(self.thetas)
            return loss_elem + regularization_term
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)
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
            m, _ = y.shape
            loss = float((((y_hat - y).T.dot(y_hat - y)) / (2 * m))[0][0])
            regularization_term = (self.lambda_ / (2 * m)) * l2(self.thetas)
            return loss + regularization_term
        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)
            return None

    def gradient_(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Computes the regularized linear gradient of three non-empty
        numpy.ndarray. The three arrays must have compatible shapes.
        Args:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        Return:
            A numpy.ndarray, a vector of shape (n + 1) * 1, containing the
                results of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles shapes.
            None if y, x or theta or lambda_ is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # computation
            m, _ = x.shape
            x_prime = np.hstack((np.ones((m, 1)), x))
            theta_prime = self.thetas.copy()
            theta_prime[0][0] = 0
            y_hat = x_prime.dot(self.thetas)
            return (x_prime.T.dot(y_hat - y) + self.lambda_ * theta_prime) / m

        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)
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
            m, _ = x.shape
            if (y.shape[0] != m or y.shape[1] != 1):
                print('Error: wrong shape on parameter(s)', file=sys.stderr)
                return None
            # calculation of the gradient vector
            # 1. X to X'
            x_prime = np.c_[np.ones((m, 1)), x]
            # 2. loop
            for _ in range(self.max_iter):
                # 3. calculate the grandient for current thetas
                gradient = (x_prime.T.dot(x_prime.dot(self.thetas) - y) / m)
                # 4. calculate and assign the new thetas
                self.thetas -= self.alpha * gradient
            return self.thetas

        except (ValueError, TypeError, AttributeError) as exc:
            print(exc, file=sys.stderr)
            return None

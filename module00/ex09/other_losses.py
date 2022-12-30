"""Loss functions module"""
from math import sqrt
import numpy as np


def mse_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        # type test
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            print('Error: wrong type')
            return None
        # shape test
        y = y.reshape((-1, 1))
        y_hat = y_hat.reshape((-1, 1))
        if y.shape[1] != 1 or y.shape != y_hat.shape:
            print('Error: wrong shape on parameter(s)')
            return None
        # calculation
        return float((((y_hat - y).T.dot(y_hat - y)) / len(y))[0][0])
    except (TypeError, ValueError) as exc:
        print(exc)
        return None

def rmse_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        # calculation
        return sqrt(mse_(y, y_hat))
    except (TypeError, ValueError) as exc:
        print(exc)
        return None

    
def mae_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        # type test
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            print('Error: wrong type')
            return None
        # shape test
        y = y.reshape((-1, 1))
        y_hat = y_hat.reshape((-1, 1))
        if y.shape[1] != 1 or y.shape != y_hat.shape:
            print('Error: wrong shape on parameter(s)')
            return None
        # calculation
        return float(np.sum(np.absolute(y_hat - y))/ len(y))
    except (TypeError, ValueError) as exc:
        print(exc)
        return None


def r2score_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Description:
        Calculate the R2score between the predicted output and the output.
        R² = 1 - (SSE / SST)
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        # type test
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            print('Error: wrong type')
            return None
        # shape test
        y = y.reshape((-1, 1))
        y_hat = y_hat.reshape((-1, 1))
        if y.shape[1] != 1 or y.shape != y_hat.shape:
            print('Error: wrong shape on parameter(s)')
            return None
        # calculation
        mean_y = np.mean(y)
        sse = float((((y_hat - y).T.dot(y_hat - y)))[0][0])
        sst = float((((y - mean_y).T.dot(y - mean_y)))[0][0])
        return 1 - (sse / sst)
    except (TypeError, ValueError) as exc:
        print(exc)
        return None


if __name__ == "__main__":

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Example 1:
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])

    # Mean squared error
    assert mse_(x,y) == mean_squared_error(x,y)
    ## your implementation
    assert mse_(x,y) == 4.285714285714286
    ## sklearn implementation
    assert mean_squared_error(x,y) == 4.285714285714286
    
    # Root mean squared error
    assert rmse_(x,y) == sqrt(mean_squared_error(x,y))
    ## your implementation
    assert rmse_(x,y) == 2.0701966780270626
    ## sklearn implementation not available: take the square root of MSE
    assert sqrt(mean_squared_error(x,y)) == 2.0701966780270626

    # Mean absolute error
    assert mae_(x,y) == mean_absolute_error(x,y)
    ## your implementation
    assert mae_(x,y) == 1.7142857142857142
    ## sklearn implementation
    assert mean_absolute_error(x,y) == 1.7142857142857142

    # R2-score
    assert r2score_(x,y) == r2_score(x,y)
    ## your implementation
    assert r2score_(x,y) == 0.9681528662420382
    ## sklearn implementation
    assert r2_score(x,y) == 0.9681528662420382

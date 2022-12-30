"""Loss function module"""
import numpy as np


def loss_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training
            examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # type test
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            print('Error: wrong type')
            return None
        # shape test
        if (y.shape != y_hat.shape):
            print('Error: wrong shape on parameter(s)')
            return None
        return (y_hat - y) ** 2

    except (TypeError, ValueError) as exc:
        print(exc)
        return None


def loss_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Description:
    Calculates the value of loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        loss_vector = loss_elem_(y, y_hat)
        return np.sum(loss_vector) / (2 * len(y))

    except (TypeError, ValueError) as exc:
        print(exc)
        return None


if __name__ == "__main__":

    # Import predict function
    import sys
    sys.path.insert(1, '../ex04/')
    from prediction import predict_

    # prepare test data
    x1 = np.array([[0.],
                   [1.],
                   [2.],
                   [3.],
                   [4.]]).reshape((-1, 1))

    theta1 = np.array([[2.],
                       [4.]]).reshape((-1, 1))

    y_hat1 = predict_(x1, theta1)

    y1 = np.array([[2.],
                   [7.],
                   [12.],
                   [17.],
                   [22.]]).reshape((-1, 1))

    # Example 1:
    print(loss_elem_(y1, y_hat1))
    np.testing.assert_array_equal(loss_elem_(y1, y_hat1),np.array([[0.],
                                                                   [ 1],
                                                                   [ 4],
                                                                   [ 9],
                                                                   [16]]))

    # Example 2:
    assert(loss_(y1, y_hat1) == 3.0)
    
    # prepare test data
    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

    # Example 3:
    assert(loss_(y2, y_hat2) == 2.142857142857143)
    
    # Example 4:
    assert(loss_(y2, y2) == 0.0)

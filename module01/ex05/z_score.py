"""Z-score standardization"""
import math
import numpy as np


# specific function to get the SAMPLE standard deviation
# def var_(x: np.ndarray) -> float:
#     """computes the variance of a given non-empty list or array x"""
#     result = 0
#     for data in x:
#         result += (data - np.mean(x)) ** 2
#     return float(result / (len(x)))

# def std_(x: np.ndarray) -> float:
#     """ computes the standard deviation of a given non-empty list or
#     array x"""
#     return math.sqrt(var_(x))

def zscore(x: np.ndarray) -> np.ndarray:
    """
    Computes the normalized version of a non-empty numpy.ndarray using the
        z-score standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x' as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn't raise any Exception.
    """
    try:
        # type test
        if not isinstance(x, np.ndarray):
            print("Something went wrong")
            return None
        # shape test
        x = x.reshape((-1, 1))
        # normalization
        z_score_formula = lambda x, std, mean: (x - mean) / std
        zscore_normalize = np.vectorize(z_score_formula)
        x_prime = zscore_normalize(x, np.std(x), np.mean(x))
        return x_prime

    except (ValueError, TypeError, AttributeError) as exc:
        print(exc)
        return None


if __name__ == "__main__":

    # Example 1:
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(zscore(X))
    # output : np.array([-0.08620324,
    #                     1.2068453 ,
    #                    -0.86203236,
    #                     0.51721942,
    #                     0.94823559,
    #                     0.17240647,
    #                    -1.89647119])

    # Example 2:
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(zscore(Y))
    # output : np.array([0.11267619,
    #                    1.16432067,
    #                   -1.20187941,
    #                    0.37558731,
    #                    0.98904659,
    #                    0.28795027,
    #                   -1.72770165])

    # The z-score formula is :
    #
    # get the difference between one point an the mean then divide it by the
    #  standard deviation so you will have instead of the data itself, a number
    #  correlated to how many times this data is far away from the mean, in
    #  standard deviation unit.
    # As the standard deviation will increase if there is a lot of data far
    #  away from the mean, this number will decrease. So it allow us to keep
    #  the data in a certain range

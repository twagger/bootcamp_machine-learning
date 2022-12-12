"""Initiation to very basic statistic notions."""
from typing import Union
from typing import List
import math
import numpy as np

Vector = List[Union[int, float, complex]]


class TinyStatistician():
    """Tiny Statistician"""

    def __init__(self):
        """Constructor"""
        pass

    def mean(self, x: Vector) -> float:
        """Computes the mean of a given non-empty list or array x"""
        # Emptyness and type check
        if (not isinstance(x, (list, np.ndarray))
            or (isinstance(x, (list, np.ndarray)) and len(x) == 0)):
            print(f'Error: x should be a vector of numeric values: {x}')
            return None
        # Computation
        result: float = 0
        try:
            for num in x:
                result += num
            return float(result / len(x))
        except TypeError:
            print(f'Error: x should be a vector of numeric values: {x}')
            return None

    def median(self, x: Vector) -> float:
        """Computes the median of a given non-empty list or array x"""
        return float(self.percentile(x, 50))
    
    def percentile(self, x: Vector, p: int) -> float:
        """
        computes the expected percentile of a given non-empty list or array x.
        """
        # Emptyness and type check
        if (not isinstance(x, (list, np.ndarray))
            or not isinstance(p, int)
            or (isinstance(p, int) and (p < 0 or p > 100))
            or (isinstance(x, (list, np.ndarray)) and len(x) == 0)):
            print(f'Error: x should be a vector of numeric val + 1ues: {x}'
                  f' and p should be an int between 0 and 100: {p}')
            return None
        # Computation
        x.sort()
        # specific values
        try:
            if p in (0, 100):
                return x[len(x) - 1] if p == 100 else x[0]
            fractional_rank: float = (p / 100) * (len(x) - 1)
            int_part = int(fractional_rank)
            frac_part = fractional_rank % 1
            return (x[int_part] + frac_part * (x[int_part + 1] - x[int_part]))
        except TypeError:
            print(f'Error: x should be a vector of numeric values: {x}')
            return None

    def quartiles(self, x: Vector) -> Vector:
        """Computes the 1st and 3rd quartiles of a given non-empty array x"""
        # Emptyness and type check
        if (not isinstance(x, (list, np.ndarray))
            or (isinstance(x, (list, np.ndarray)) and len(x) == 0)):
            print(f'Error: x should be a vector of numeric values: {x}')
            return None
        # Computation
        return ([float(self.percentile(x, 25)), float(self.percentile(x, 75))])

    def var(self, x: Vector) -> float:
        """computes the variance of a given non-empty list or array x"""
        # Emptyness and type check
        if (not isinstance(x, (list, np.ndarray))
            or (isinstance(x, (list, np.ndarray)) and len(x) == 0)):
            print(f'Error: x should be a vector of numeric values: {x}')
            return None
        # Computation
        result = 0
        try:
            for num in x:
                result += (num - self.mean(x)) ** 2
            return float(result / (len(x) - 1))
        except TypeError:
            print(f'Error: x should be a vector of numeric values: {x}')
            return None


    def std(self, x: Vector) -> float:
        """ computes the standard deviation of a given non-empty list or
        array x"""
        # Emptyness check
        if isinstance(x, (list, np.ndarray)) and len(x) == 0:
            return None
        # Computation
        return math.sqrt(self.var(x))


if __name__ == "__main__":

    ts = TinyStatistician()

    # mean
    print("\033[1;35m--mean--\033[0m")
    print("\033[33mvector = [1, 42, 300, 10, 59]\033[0m")
    vector = [1, 42, 300, 10, 59]
    print(f'\033[1;35m>\033[0m {ts.mean(vector)}')

    print("\n\033[1;35m--mean on non vector--\033[0m")
    print("\033[33mvector = [[1, 2, 3, 4, 5], [1, 2, 3]]\033[0m")
    vector = [[1, 2, 3, 4, 5], [1, 2, 3]]
    print(f'\033[1;35m>\033[0m {ts.mean(vector)}')

    print("\n\033[1;35m--mean on non vector--\033[0m")
    print("\033[33mvector = \"This is wrong\"\033[0m")
    vector = "This is wrong"
    print(f'\033[1;35m>\033[0m {ts.mean(vector)}')

    # median
    print("\n\033[1;35m--median--\033[0m")
    print("\033[33mvector = [1, 42, 300, 10, 59]\033[0m")
    vector = [1, 42, 300, 10, 59]
    print(f'\033[1;35m>\033[0m {ts.median(vector)}')

    print("\n\033[1;35m--median of 10 elements--\033[0m")
    print("\033[33mvector = [663.03, 816.77, 689.50, 726.54, 800.53, 741.16, "
          "738.93, 788.65, 824.50, 732.59]\033[0m")
    vector = [663.03, 816.77, 689.50, 726.54, 800.53, 741.16, 738.93, 788.65,
              824.50, 732.59]
    print(f'\033[1;35m>\033[0m {ts.median(vector)}')

    print("\n\033[1;35m--median of 9 elements--\033[0m")
    print("\033[33mvector = [663.03, 816.77, 689.50, 726.54, 800.53, 741.16, "
          "738.93, 788.65, 824.50]\033[0m")
    vector = [663.03, 816.77, 689.50, 726.54, 800.53, 741.16, 738.93, 788.65,
              824.50]
    print(f'\033[1;35m>\033[0m {ts.median(vector)}')

    # percentile
    print("\n\033[1;35m--percentile--\033[0m")
    print("\033[33mvector = [1, 42, 300, 10, 59] / percentile = 10\033[0m")
    vector = [1, 42, 300, 10, 59]
    print(f'\033[1;35m>\033[0m {ts.percentile(vector, 10)}')

    print("\n\033[1;35m--percentile--\033[0m")
    print("\033[33mvector = [1, 42, 300, 10, 59] / percentile = 15\033[0m")
    vector = [1, 42, 300, 10, 59]
    print(f'\033[1;35m>\033[0m {ts.percentile(vector, 15)}')

    print("\n\033[1;35m--percentile--\033[0m")
    print("\033[33mvector = [1, 42, 300, 10, 59] / percentile = 20\033[0m")
    vector = [1, 42, 300, 10, 59]
    print(f'\033[1;35m>\033[0m {ts.percentile(vector, 20)}')

    # quartile
    print("\n\033[1;35m--quartile--\033[0m")
    print("\033[33mvector = [1, 42, 300, 10, 59]\033[0m")
    vector = [1, 42, 300, 10, 59]
    print(f'\033[1;35m>\033[0m {ts.quartiles(vector)}')

    # variance
    print("\n\033[1;35m--variance--\033[0m")
    print("\033[33mvector = [1, 42, 300, 10, 59]\033[0m")
    vector = [1, 42, 300, 10, 59]
    print(f'\033[1;35m>\033[0m {ts.var(vector)}')

    # standard deviation
    print("\n\033[1;35m--standard deviation--\033[0m")
    print("\033[33mvector = [1, 42, 300, 10, 59]\033[0m")
    vector = [1, 42, 300, 10, 59]
    print(f'\033[1;35m>\033[0m {ts.std(vector)}')
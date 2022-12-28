"""Matrix module"""


class Matrix():
    """Matrix class"""

    def __init__(self, data):
        """Constructor"""
        # list constructor
        if isinstance(data, list):
            m = len(data) if data is not None else 0
            n = len(data[0]) if data is not None else 0
            # check if all rows have the same length
            if not all(len(row) == n for row in data):
                raise ValueError(f'Wrong shape on value : {data}')
            self.data = data
            self.shape = (m, n)
        # shape constructor
        elif isinstance(data, tuple):
            # check if tuple is ok
            if len(data) != 2 or data is None:
                raise ValueError(f'Wrong shape: {data}')
            if not isinstance(data[0], int) or not isinstance(data[1], int):
                raise TypeError(f'Wrong type: {data}')
            if data[0] < 0 or data[1] < 0:
                raise ValueError(f'Dimensions of a shape must be positives')
            # creation of the matrix
            self.data = [[0 for n in range(data[1])] for m in range(data[0])]
            self.shape = (data[0], data[1])
        else:
            raise TypeError(f'Wrong type : {type(data)}')

    # Operators overload
    def __add__(self, rhs):
        """add: only matrices of same dimensions."""
        if rhs.shape == self.shape:
            try:
                result = [[self.data[m][n] + rhs.data[m][n]
                        for n in range(self.shape[1])]
                        for m in range(self.shape[0])]
            except TypeError:
                result = [self.data[n] - rhs.data[n]
                          for n in range(self.shape[1])]
            return type(self)(result)
        raise AttributeError("Shape error")

    def __radd__(self, rhs):
        """radd: only matrices of same dimensions."""
        return rhs + self

    def __sub__(self, rhs):
        """sub: only matrices of same dimensions."""
        if rhs.shape == self.shape:
            try:
                result = [[self.data[m][n] - rhs.data[m][n]
                        for n in range(self.shape[1])]
                        for m in range(self.shape[0])]
            except TypeError:
                result = [self.data[n] - rhs.data[n]
                          for n in range(self.shape[1])]
            return type(self)(result)
        raise AttributeError("Shape error")

    def __rsub__(self, rhs):
        """rsub: only matrices of same dimensions."""
        return rhs - self

    def __truediv__(self, scalar):
        """div: only scalars."""
        if scalar == 0:
            raise ArithmeticError("Division by 0.")
        try:
            result = [[self.data[m][n] / scalar for n in range(self.shape[1])]
                    for m in range(self.shape[0])]
        except TypeError:
            result = [self.data[n] / scalar for n in range(self.shape[1])]
        return type(self)(result)

    def __rtruediv__(self, scalar):
        """rdiv: not implemented."""
        raise ArithmeticError("Division of a scalar by a Matrix / Vector"
                              " is not defined here.")

    def __mul__(self, rhs):
        """mul: scalars, vectors and matrixes."""
        try:
            if isinstance(rhs, (int, float, complex)):
                try:
                    result = [[self.data[m][n] * rhs
                               for n in range(self.shape[1])]
                              for m in range(self.shape[0])]
                except TypeError:
                    result = [self.data[m] * rhs for m in range(self.shape[0])]
                return type(self)(result)

            elif type(rhs) == Matrix and type(self) == Matrix:
                # verify shape compatibility
                if self.shape[1] != rhs.shape[0]:
                    raise AttributeError("Shapes are not compatibles")
                # compute result
                result = [[sum(self.data[m][n] * rhs.data[n][p]
                            for n in range(self.shape[1]))
                           for p in range(rhs.shape[1])]
                          for m in range(self.shape[0])]
                return type(self)(result)

            elif type(rhs) == Vector and type(self) == Matrix:
                # verify shape compatibility
                if self.shape[1] != rhs.shape[0]:
                    raise AttributeError("Shapes are not compatibles")
                # compute result
                result = [sum(self.data[m][n] * rhs.data[n][0]
                               for n in range(self.shape[1]))
                          for m in range(self.shape[0])]
                return Vector(result)
            else:
                raise TypeError(f'Wrong type: {rhs}')

        except (TypeError, AttributeError, ValueError):
            raise TypeError("Error")

    def __rmul__(self, rhs):
        """rmul: scalar, vectors and matrixes."""
        try:
            if isinstance(rhs, (int, float, complex)):
                try:
                    result = [[self.data[m][n] * rhs
                               for n in range(self.shape[1])]
                              for m in range(self.shape[0])]
                except TypeError:
                    result = [self.data[m] * rhs for m in range(self.shape[0])]
                return type(self)(result)

            elif type(rhs) == Matrix and type(self) == Matrix:
                # verify shape compatibility
                if rhs.shape[1] != self.shape[0]:
                    raise AttributeError("Shapes are not compatibles")
                # compute result
                result = [[sum(rhs.data[m][n] * self.data[n][p]
                            for n in range(self.shape[1]))
                           for p in range(rhs.shape[1])]
                          for m in range(self.shape[0])]
                return type(self)(result)

            elif type(self) == Vector and type(rhs) == Matrix:
                # verify shape compatibility
                return rhs * self

        except (TypeError, AttributeError, ValueError):
            raise TypeError("Error")

    def T(self):
        """Transpose matrix (row / column)"""
        if type(self) == Matrix:
            return Matrix([[self.data[m][n] for m in range(self.shape[0])]
                           for n in range(self.shape[1])])
        elif type(self) == Vector:
            if self.shape[0] == 1:
                return Vector([[self.data[n]] for n in range(self.shape[1])])
            elif self.shape[1] == 1:
                return Vector([self.data[m][0] for m in range(self.shape[0])])

    def __str__(self):
        return f'{self.data}'

    def __repr__(self):
        return f'{self.data}'


class Vector(Matrix):
    """Vector class"""

    def __init__(self, data):
        """Constructor"""
        # list constructor
        if isinstance(data, list):
            n = len(data) if data is not None else 0
            # row vector
            if any(isinstance(item, list) for item in data):
                try:
                    if not all(len(row) == 1 for row in data):
                        raise ValueError(f'Wrong shape : {data}')
                except TypeError:
                    raise ValueError(f'Wrong shape: {data}')
                if not all([isinstance(value, (int, float, complex))
                            for value in row] for row in data):
                    raise TypeError(f'Wrong type : {data}')
                self.data = data
                self.shape = (n, 1 if len(data) != 0 else 0)
            # column vector
            else:
                if not all(isinstance(value, (int, float, complex))
                           for value in data):
                    raise TypeError(f'Wrong type : {data}')
                self.data = data
                self.shape = (1 if len(data) != 0 else 0, n)
        else:
            raise TypeError(f'Wrong type: {data}')

    def dot(self, v):
        """Dot product"""
        if v.shape == self.shape:
            if self.shape[1] == 1:
                return sum(self.data[i][0] * v.data[i][0]
                           for i in range(len(self.data)))
            return sum(self.data[0][i] * v.data[0][i]
                       for i in range(len(self.data[0])))
        raise AttributeError("Vectors don't have the same shape")

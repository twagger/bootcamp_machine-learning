"""Test module"""
from matrix import Matrix
from matrix import Vector


if __name__ == '__main__':

    # -----------------
    # CONSTRUCTOR TESTS
    # -----------------
    print(f'{"":-<60}')

    # Creation from list
    mat = Matrix([[1, 2, 3], [4, 5, 6]])
    print(f'{"From list":20}: {mat}')
    print(f'{"":20}: {mat.shape}')

    # Creation with empty list
    mat = Matrix([[], []])
    print(f'{"Empty list":20}: {mat}')
    print(f'{"":20}: {mat.shape}')

    # From non square list
    try:
        mat = Matrix([[1, 2, 3], [4, 5, 6], [7, 8]])
    except ValueError as exc:
        print(f'{"Non square list":20}: {exc}')

    # From tuple
    mat = Matrix((3, 2))
    print(f'{"From tuple":20}: {mat}')
    print(f'{"":20}: {mat.shape}')

    # From tuple with 0
    mat = Matrix((0, 3))
    print(f'{"From tuple with 0":20}: {mat}')
    print(f'{"":20}: {mat.shape}')

    # From tuple with 1
    mat = Matrix((3, 1))
    print(f'{"From tuple with 1":20}: {mat}')
    print(f'{"":20}: {mat.shape}')

    # From tuple non int
    try:
        mat = Matrix((2.1, 3))
    except TypeError as exc:
        print(f'{"From tuple non int":20}: {exc}')

    # -----------------
    # ADD
    # -----------------
    print(f'{"":-<60}')

    # simple addition
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    m2 = Matrix([[4, 4, 4], [3, 3, 3], [2, 2, 2]])
    result = m1 + m2
    print(f'{"Addition":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # wrong addition
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    m2 = Matrix([[4, 4], [3, 3], [2, 2]])
    try:
        result = m1 + m2
    except AttributeError as exc:
        print(f'{"Addition":20}: {exc}')

    # -----------------
    # SUB
    # -----------------
    print(f'{"":-<60}')

    # simple substraction
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    m2 = Matrix([[4, 4, 4], [3, 3, 3], [2, 2, 2]])
    result = m1 - m2
    print(f'{"Substraction":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # wrong substraction
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    m2 = Matrix([[4, 4], [3, 3], [2, 2]])
    try:
        result = m1 - m2
    except AttributeError as exc:
        print(f'{"Substraction":20}: {exc}')

    # simple substraction
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    m2 = Matrix([[4, 4, 4], [3, 3, 3], [2, 2, 2]])
    result = m1.__rsub__(m2)
    print(f'{"Substraction":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # -----------------
    # DIV
    # -----------------
    print(f'{"":-<60}')

    # simple division
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    result = m1 / 2
    print(f'{"Division":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # wrong division
    try:
        result = m1 / 0
    except ArithmeticError as exc:
        print(f'{"Division by 0":20}: {exc}')

    # wrong division
    try:
        result = 8 / m1
    except ArithmeticError as exc:
        print(f'{"Division of a scalar":20}: {exc}')

    # -----------------
    # MUL
    # -----------------
    print(f'{"":-<60}')

    # scalar multiplication
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    result = m1 * 2
    print(f'{"Scalar mul":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # vector multiplication
    m1 = Matrix([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    v1 = Vector([[1], [2], [3]])
    result = m1 * v1
    print(f'{"Vector mul":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # matrix multiplication
    m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                 [0.0, 2.0, 4.0, 6.0]])
    m2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    result = m1 * m2
    print(f'{"Matrix mul":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # rev scalar multiplication
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    result = m1.__rmul__(2)
    print(f'{"Scalar rmul":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # -----------------
    # TRANSPOSITION
    # -----------------
    m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                 [0.0, 2.0, 4.0, 6.0]])
    print(f'{"Transposed":20}: {m1.T()}')
    print(f'{"":20}: {m1.T().shape}')

    # -----------------
    # VECTOR CONSTRUCTOR TESTS
    # -----------------
    print(f'{"":-<60}')

    # Creation row vector from list
    vec = Vector([1, 2, 3])
    print(f'{"From list":20}: {vec}')
    print(f'{"":20}: {vec.shape}')

    # Creation column vector from list
    vec = Vector([[1], [2], [3]])
    print(f'{"Empty list":20}: {vec}')
    print(f'{"":20}: {vec.shape}')

    # Creation with empty list
    vec = Vector([])
    print(f'{"Empty list":20}: {vec}')
    print(f'{"":20}: {vec.shape}')

    # From non scalar list
    try:
        vec = Vector(['toto', 'titi', 'tata'])
    except TypeError as exc:
        print(f'{"Non scalar list":20}: {exc}')

    # Creation with wrong list
    try:
        vec = Vector([[1], [2], [3]])
    except (TypeError, ValueError) as exc:
        print(f'{"Wrong list":20}: {exc}')

    # -----------------
    # VECTOR ADD
    # -----------------
    print(f'{"":-<60}')

    # simple addition
    v1 = Vector([2, 2, 2])
    v2 = Vector([4, 4, 4])
    result = v1 + v2
    print(f'{"Addition":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # wrong addition
    v1 = Vector([2, 2, 2])
    v2 = Vector([4, 4])
    try:
        result = v1 + v2
    except AttributeError as exc:
        print(f'{"Addition":20}: {exc}')

    # -----------------
    # VECTOR SUB
    # -----------------
    print(f'{"":-<60}')

    # simple substraction
    v1 = Vector([2, 2, 2])
    v2 = Vector([4, 4, 4])
    result = v1 - v2
    print(f'{"Substraction":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # simple substraction 2
    v1 = Vector([[2], [2], [2]])
    v2 = Vector([[4], [4], [4]])
    result = v1 - v2
    print(f'{"Substraction":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # wrong substraction
    v1 = Vector([2, 2, 2, 2])
    v2 = Vector([4, 4])
    try:
        result = v1 - v2
    except AttributeError as exc:
        print(f'{"Substraction":20}: {exc}')

    # simple rsubstraction 
    v1 = Vector([2, 2, 2])
    v2 = Vector([4, 4, 4])
    result = v1.__rsub__(v2)
    print(f'{"rSubstraction":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # simple rsubstraction 2
    v1 = Vector([[2], [2], [2]])
    v2 = Vector([[4], [4], [4]])
    result = v1.__rsub__(v2)
    print(f'{"rSubstraction":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # -----------------
    # VECTOR DIV
    # -----------------
    print(f'{"":-<60}')

    # simple division
    v1 = Vector([2, 2, 2, 3])
    result = v1 / 2
    print(f'{"Division":20}: {result}')
    print(f'{"":20}: {result.shape}')

    # wrong division
    try:
        result = v1 / 0
    except ArithmeticError as exc:
        print(f'{"Division by 0":20}: {exc}')

    # wrong division
    try:
        result = 8 / v1
    except ArithmeticError as exc:
        print(f'{"Division of a scalar":20}: {exc}')

    # -----------------
    # VECTOR TRANSPOSITION
    # -----------------
    v1 = Vector([0.0, 1.0, 2.0, 3.0])

    print(f'{"Transposed":20}: {v1.T()}')
    print(f'{"":20}: {v1.T().shape}')

    v1 = Vector([[0.0], [1.0], [2.0], [3.0]])

    print(f'{"Transposed":20}: {v1.T()}')
    print(f'{"":20}: {v1.T().shape}')
"""Test module"""
from matrix import Matrix
from matrix import Vector


if __name__ == '__main__':

    # -----------------
    # TESTS FROM SUBJECT
    # -----------------
    print(f'{"":-<60}')
    print("\033[1;35mSubject tests\033[0m")
    print(f'{"":-<60}')

    m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(f'{"m1.shape":20}: {m1.shape}')
    # Output:
    #(3, 2)

    print(f'{"m1.T()":20}: {m1.T().__class__.__name__}({m1.T()})')
    # Output:
    # Matrix([[0., 2., 4.], [1., 3., 5.]])

    print(f'{"m1.T().shape":20}: {m1.T().shape}')
    # Output:
    # (2, 3)

    m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
    print(f'{"m1.shape":20}: {m1.shape}')
    # Output:
    # (2, 3)

    print(f'{"m1.T()":20}: {m1.T().__class__.__name__}({m1.T()})')
    # Output:
    # Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    print(f'{"m1.T().shape":20}: {m1.T().shape}')
    # Output:
    # (3, 2)

    m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                 [0.0, 2.0, 4.0, 6.0]])
    m2 = Matrix([[0.0, 1.0],
                 [2.0, 3.0],
                 [4.0, 5.0],
                 [6.0, 7.0]])
    print(f'{"m1 * m2":20}: {(m1 * m2).__class__.__name__}({(m1 * m2)})')
    # Output:
    # Matrix([[28., 34.], [56., 68.]])

    m1 = Matrix([[0.0, 1.0, 2.0],
                 [0.0, 2.0, 4.0]])
    v1 = Vector([[1], [2], [3]])
    print(f'{"m1 * v1":20}: {(m1 * v1).__class__.__name__}({m1 * v1})')
    # Output:
    # Matrix([[8], [16]])
    # Or: Vector([[8], [16]

    v1 = Vector([[1], [2], [3]])
    v2 = Vector([[2], [4], [8]])
    print(f'{"v1 + v2":20}: {(v1 + v2).__class__.__name__}({v1 + v2})')
    # Output:
    # Vector([[3],[6],[11]])

    # -----------------
    # CONSTRUCTOR TESTS
    # -----------------
    print(f'{"":-<60}')
    print("\033[1;35mMatrix constructor\033[0m")
    print(f'{"":-<60}')

    # Creation from list
    mat = Matrix([[1, 2, 3], [4, 5, 6]])
    print(f'{"From list":20}: {mat}')

    # Creation with empty list
    mat = Matrix([[], []])
    print(f'{"Empty list":20}: {mat}')

    # From non square list
    try:
        mat = Matrix([[1, 2, 3], [4, 5, 6], [7, 8]])
    except ValueError as exc:
        print(f'{"Non square list":20}: {exc}')

    # From tuple
    mat = Matrix((3, 2))
    print(f'{"From tuple":20}: {mat}')

    # From tuple with 0
    mat = Matrix((0, 3))
    print(f'{"From tuple with 0":20}: {mat}')

    # From tuple with 1
    mat = Matrix((3, 1))
    print(f'{"From tuple with 1":20}: {mat}')

    # From tuple non int
    try:
        mat = Matrix((2.1, 3))
    except TypeError as exc:
        print(f'{"From tuple non int":20}: {exc}')

    # -----------------
    # ADD
    # -----------------
    print(f'{"":-<60}')
    print("\033[1;35mMatrix add\033[0m")
    print(f'{"":-<60}')

    # simple addition
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    m2 = Matrix([[4, 4, 4], [3, 3, 3], [2, 2, 2]])
    result = m1 + m2
    print(f'{"Addition":20}: {result}')

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
    print("\033[1;35mMatrix sub\033[0m")
    print(f'{"":-<60}')

    # simple substraction
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    m2 = Matrix([[4, 4, 4], [3, 3, 3], [2, 2, 2]])
    result = m1 - m2
    print(f'{"Substraction":20}: {result}')

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

    # -----------------
    # DIV
    # -----------------
    print(f'{"":-<60}')
    print("\033[1;35mMatrix div\033[0m")
    print(f'{"":-<60}')

    # simple division
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    result = m1 / 2
    print(f'{"Division":20}: {result}')

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
    print("\033[1;35mMatrix mul\033[0m")
    print(f'{"":-<60}')

    # scalar multiplication
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    result = m1 * 2
    print(f'{"Scalar mul":20}: {result}')

    # vector multiplication
    m1 = Matrix([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    v1 = Vector([[1], [2], [3]])
    result = m1 * v1
    print(f'{"Vector mul":20}: {result}')

    # matrix multiplication
    m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                 [0.0, 2.0, 4.0, 6.0]])
    m2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    result = m1 * m2
    print(f'{"Matrix mul":20}: {result}')

    # rev scalar multiplication
    m1 = Matrix([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    result = m1.__rmul__(2)
    print(f'{"Scalar rmul":20}: {result}')

    # -----------------
    # TRANSPOSITION
    # -----------------
    print(f'{"":-<60}')
    print("\033[1;35mMatrix transposition\033[0m")
    print(f'{"":-<60}')
    m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                 [0.0, 2.0, 4.0, 6.0]])
    print(f'{"Transposed":20}: {m1.T()}')

    # -----------------
    # VECTOR CONSTRUCTOR TESTS
    # -----------------
    print(f'{"":-<60}')
    print("\033[1;35mVector constructor\033[0m")
    print(f'{"":-<60}')

    # Creation row vector from list
    vec = Vector([1, 2, 3])
    print(f'{"From list":20}: {vec}')

    # Creation column vector from list
    vec = Vector([[1], [2], [3]])
    print(f'{"Empty list":20}: {vec}')

    # Creation with empty list
    vec = Vector([])
    print(f'{"Empty list":20}: {vec}')

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
    print("\033[1;35mVector add\033[0m")
    print(f'{"":-<60}')

    # simple addition
    v1 = Vector([2, 2, 2])
    v2 = Vector([4, 4, 4])
    result = v1 + v2
    print(f'{"Addition":20}: {result}')

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
    print("\033[1;35mVector sub\033[0m")
    print(f'{"":-<60}')

    # simple substraction
    v1 = Vector([2, 2, 2])
    v2 = Vector([4, 4, 4])
    result = v1 - v2
    print(f'{"Substraction":20}: {result}')

    # simple substraction 2
    v1 = Vector([[2], [2], [2]])
    v2 = Vector([[4], [4], [4]])
    result = v1 - v2
    print(f'{"Substraction":20}: {result}')

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

    # simple rsubstraction 2
    v1 = Vector([[2], [2], [2]])
    v2 = Vector([[4], [4], [4]])
    result = v1.__rsub__(v2)
    print(f'{"rSubstraction":20}: {result}')

    # -----------------
    # VECTOR DIV
    # -----------------
    print(f'{"":-<60}')
    print("\033[1;35mVector div\033[0m")
    print(f'{"":-<60}')

    # simple division
    v1 = Vector([2, 2, 2, 3])
    result = v1 / 2
    print(f'{"Division":20}: {result}')

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
    print(f'{"":-<60}')
    print("\033[1;35mVector transposition\033[0m")
    print(f'{"":-<60}')

    v1 = Vector([0.0, 1.0, 2.0, 3.0])

    print(f'{"Transposed":20}: {v1.T()}')

    v1 = Vector([[0.0], [1.0], [2.0], [3.0]])

    print(f'{"Transposed":20}: {v1.T()}')

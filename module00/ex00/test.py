"""Test module"""
from matrix import Matrix


if __name__ == '__main__':

    print("\033[1;35m--Simple matrix creation--\033[0m")
    print("\033[33mmat = Matrix([[1,2,3], [4,5,6]])\033[0m")
    mat = Matrix([[1,2,3], [4,5,6]])
    print(f'\033[1;35m>\033[0m {mat}')

    print("\n\033[1;35m--Empty matrix creation--\033[0m")
    print("\033[33mmat = Matrix([[], []])\033[0m")
    mat = Matrix([[], []])
    print(f'\033[1;35m>\033[0m {mat}')

    print("\n\033[1;35m--Non square matrix creation--\033[0m")
    print("\033[33mmat = Matrix([[1,2,3], [4,5,6], [7,8]])\033[0m")
    mat = Matrix([[1,2,3], [4,5,6], [7,8]])
    print(f'\033[1;35m>\033[0m {mat}')
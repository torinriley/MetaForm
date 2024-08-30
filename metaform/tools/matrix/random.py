from .matrix import Matrix
import random

class MatrixRandom:
    @staticmethod
    def random(rows, cols, low=0.0, high=1.0):
        # Random matrix of given dimensions
        return Matrix([[random.uniform(low, high) for _ in range(cols)] for _ in range(rows)])

    @staticmethod
    def random_int(rows, cols, low=0, high=10):
        # Random integer matrix of given dimensions
        return Matrix([[random.randint(low, high) for _ in range(cols)] for _ in range(rows)])

from ....tools.matrix.matrix import Matrix
from ....tools.matrix.algebra import MatrixAlgebra
import math
import random
from ....tools.matrix.statistics import Statistics

class ActivationFunctions:
    @staticmethod
    def relu(matrix):
        return Matrix([[max(0, val) for val in row] for row in matrix.data])

    @staticmethod
    def sigmoid(matrix):
        return Matrix([[1 / (1 + math.exp(-val)) for val in row] for row in matrix.data])

    @staticmethod
    def tanh(matrix):
        return Matrix([[math.tanh(val) for val in row] for row in matrix.data])

class Dropout:
    def __init__(self, drop_prob):
        self.drop_prob = drop_prob

    def apply(self, matrix):
        mask = [[1 if random.random() > self.drop_prob else 0 for _ in range(matrix.cols)] for _ in range(matrix.rows)]
        return Matrix([[val * mask[i][j] for j, val in enumerate(row)] for i, row in enumerate(matrix.data)])

class MatrixNormalization:
    @staticmethod
    def layer_norm(matrix, eps=1e-5):
        mean = Statistics.mean(matrix)
        variance = Statistics.variance(matrix)
        normalized = Matrix([[(val - mean) / math.sqrt(variance + eps) for val in row] for row in matrix.data])
        return normalized

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, matrix, gradient):
        return matrix - (gradient * self.learning_rate)

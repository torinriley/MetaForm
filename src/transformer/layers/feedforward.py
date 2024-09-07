# training/feed_forward_layer.py
import random
import math
from tools.matrix import Matrix
from tools.activation import ReLU

class FeedForwardLayer:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.weights1 = self.he_initialization(input_dim, hidden_dim)
        self.biases1 = Matrix([[0.0] * hidden_dim])
        self.weights2 = self.he_initialization(hidden_dim, input_dim)
        self.biases2 = Matrix([[0.0] * input_dim])
        self.relu = ReLU()

    def he_initialization(self, in_dim, out_dim):
        stddev = math.sqrt(2 / in_dim)
        return Matrix([[self.normal_random(0, stddev) for _ in range(out_dim)] for _ in range(in_dim)])
    
    def normal_random(self, mean, stddev):
        return mean + stddev * random.gauss(0, 1)

    def forward(self, x):
       
       
        # first linear transformation (input_dim -> hidden_dim)
        x = self._matrix_multiply(x, self.weights1)
        x = self._matrix_add(x, self.biases1)
        
        # ReLU activation
        x = self.relu.forward(x)
        
        # second linear transformation (hidden_dim -> input_dim)
        x = self._matrix_multiply(x, self.weights2)
        x = self._matrix_add(x, self.biases2)
        
        return x

    def _matrix_multiply(self, matrix_a, matrix_b):
        
        # matrix multiplication n
        result = [[0] * matrix_b.cols for _ in range(matrix_a.rows)]
        for i in range(matrix_a.rows):
            for j in range(matrix_b.cols):
                result[i][j] = sum(matrix_a.data[i][k] * matrix_b.data[k][j] for k in range(matrix_a.cols))
        return Matrix(result)

    def _matrix_add(self, matrix_a, matrix_b):
        # matrix addition 
        result = [[matrix_a.data[i][j] + matrix_b.data[i][j] for j in range(matrix_a.cols)] for i in range(matrix_a.rows)]
        return Matrix(result)

from ...tools.matrix.matrix import Matrix
import math

class LayerNormalization:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def forward(self, x: Matrix, gamma: Matrix, beta: Matrix):
        # Calculate the mean and variance across the features (rows)
        mean = Matrix([[sum(row) / len(row) for row in x.data]])
        variance = Matrix([[(val - mean.data[i][0]) ** 2 for val in row] for i, row in enumerate(x.data)])
        variance = Matrix([[sum(row) / len(row) for row in variance.data]])
        
        # Normalize
        x_normalized = Matrix([[(x.data[i][j] - mean.data[i][0]) / math.sqrt(variance.data[i][0] + self.epsilon) for j in range(len(row))] for i, row in enumerate(x.data)])
        
        # Scale and shift
        y = Matrix([[gamma.data[i][j] * x_normalized.data[i][j] + beta.data[i][j] for j in range(len(row))] for i, row in enumerate(x.data)])
        
        return y

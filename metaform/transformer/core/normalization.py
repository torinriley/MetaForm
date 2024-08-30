# normalization.py

from tools.matrix import Matrix

class LayerNorm:
    def __init__(self, hidden_dim, epsilon=1e-5):
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        # initialize gamma and beta as Matrix objects with the same dimension
        self.gamma = Matrix([[1.0] * hidden_dim])  # Initialize gamma to 1s
        self.beta = Matrix([[0.0] * hidden_dim])   # Initialize beta to 0s
    
    def forward(self, x):
        # x is a Matrix object with shape (batch_size, sequence_length, hidden_dim)
        
        # Compute mean and variance
        mean_value = self.compute_mean(x)
        variance_value = self.compute_variance(x, mean_value)
        
        # normalize
        variance_value = variance_value + self.epsilon  # Avoid division by zero
        std_dev = self.compute_sqrt(variance_value)
        x_normalized = self.compute_normalize(x, mean_value, std_dev)
        
        # Apply scale (gamma) and shift (beta)
        out = self.apply_scale_shift(x_normalized)
        return out
    
    def compute_mean(self, x):
        # Compute the mean along the last axis
        batch_size, rows, cols = x.batch_size, x.rows, x.cols
        mean = []
        for b in range(batch_size):
            mean_batch = []
            for i in range(rows):
                mean_row = [sum(x.data[b][i][j] for j in range(cols)) / cols for j in range(cols)]
                mean_batch.append(mean_row)
            mean.append(mean_batch)
        return Matrix(mean)
    
    def compute_variance(self, x, mean):
        # compute variance along the last axis
        batch_size, rows, cols = x.batch_size, x.rows, x.cols
        variance = []
        for b in range(batch_size):
            variance_batch = []
            for i in range(rows):
                variance_row = [sum((x.data[b][i][j] - mean.data[b][i][j])**2 for j in range(cols)) / cols for j in range(cols)]
                variance_batch.append(variance_row)
            variance.append(variance_batch)
        return Matrix(variance)
    
    def compute_sqrt(self, variance):
        # compute the square root of the variance
        return Matrix([[value**0.5 for value in row] for row in variance.data])
    
    def compute_normalize(self, x, mean, std_dev):
        # normalize the input x
        normalized = []
        for b in range(x.batch_size):
            normalized_batch = []
            for i in range(x.rows):
                normalized_row = [(x.data[b][i][j] - mean.data[b][i][j]) / std_dev.data[b][i][j] for j in range(x.cols)]
                normalized_batch.append(normalized_row)
            normalized.append(normalized_batch)
        return Matrix(normalized)
    
    def apply_scale_shift(self, x_normalized):
        # apply gamma (scale) and beta (shift)
        scaled_shifted = []
        for b in range(x_normalized.batch_size):
            scaled_shifted_batch = []
            for i in range(x_normalized.rows):
                scaled_shifted_row = [x_normalized.data[b][i][j] * self.gamma.data[0][j] + self.beta.data[0][j] for j in range(x_normalized.cols)]
                scaled_shifted_batch.append(scaled_shifted_row)
            scaled_shifted.append(scaled_shifted_batch)
        return Matrix(scaled_shifted)
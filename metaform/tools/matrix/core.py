class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __repr__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])

    def __add__(self, other):
        # Element-wise addition of two matrices
        result = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)

    def __sub__(self, other):
        # Element-wise subtraction of two matrices
        result = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)

    def __mul__(self, other):
        # Matrix multiplication
        result = [[sum(self.data[i][k] * other.data[k][j] for k in range(self.cols)) for j in range(other.cols)] for i in range(self.rows)]
        return Matrix(result)

    def transpose(self):
        # Transpose of the matrix
        result = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(result)

    @staticmethod
    def identity(size):
        # Identity matrix of given size
        return Matrix([[1 if i == j else 0 for j in range(size)] for i in range(size)])

    @staticmethod
    def zeros(rows, cols):
        # Zero matrix of given dimensions
        return Matrix([[0 for _ in range(cols)] for _ in range(rows)])

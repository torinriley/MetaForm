class Matrix:
    def __init__(self, data):
        if isinstance(data[0][0], list):
            self.data = data
            self.batch_size = len(data)
            self.rows = len(data[0])
            self.cols = len(data[0][0])
        else:
            self.data = [data]
            self.batch_size = 1
            self.rows = len(data)
            self.cols = len(data[0])

    def __repr__(self):
        return '\n'.join(['\n'.join([' '.join(map(str, row)) for row in batch]) for batch in self.data])

    def __add__(self, other):
        if self.batch_size != other.batch_size or self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions and batch size for addition.")
        result = [[[self.data[b][i][j] + other.data[b][i][j] for j in range(self.cols)] for i in range(self.rows)] for b in range(self.batch_size)]
        return Matrix(result)

    def __sub__(self, other):
        if self.batch_size != other.batch_size or self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions and batch size for subtraction.")
        result = [[[self.data[b][i][j] - other.data[b][i][j] for j in range(self.cols)] for i in range(self.rows)] for b in range(self.batch_size)]
        return Matrix(result)

    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrices must have appropriate dimensions for multiplication.")
        result = [[[sum(self.data[b][i][k] * other.data[b][k][j] for k in range(self.cols)) for j in range(other.cols)] for i in range(self.rows)] for b in range(self.batch_size)]
        return Matrix(result)

    def transpose(self):
        result = [[[self.data[b][j][i] for j in range(self.rows)] for i in range(self.cols)] for b in range(self.batch_size)]
        return Matrix(result)

    @staticmethod
    def identity(size, batch_size=1):
        return Matrix([[[1 if i == j else 0 for j in range(size)] for i in range(size)] for _ in range(batch_size)])

    @staticmethod
    def zeros(rows, cols, batch_size=1):
        return Matrix([[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(batch_size)])

    @staticmethod
    def from_flat_list(flat_list, rows, cols, batch_size=1):
        if len(flat_list) != batch_size * rows * cols:
            raise ValueError("The number of elements does not match the specified dimensions.")
        return Matrix([flat_list[b * rows * cols:(b + 1) * rows * cols].reshape((rows, cols)) for b in range(batch_size)])

    def shape(self):
        """
        Returns the shape of the matrix.

        Returns:
            tuple: The shape of the matrix as (batch_size, rows, cols).
        """
        return (self.batch_size, self.rows, self.cols)

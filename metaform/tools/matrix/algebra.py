from .matrix import Matrix
class MatrixAlgebra:
    @staticmethod
    def determinant(matrix):
        # Base case for 2x2 matrix
        if len(matrix.data) == 2 and len(matrix.data[0]) == 2:
            return matrix.data[0][0] * matrix.data[1][1] - matrix.data[0][1] * matrix.data[1][0]
        
        # Recursive case for larger matrices
        determinant = 0
        for c in range(len(matrix.data)):
            determinant += ((-1)**c) * matrix.data[0][c] * MatrixAlgebra.determinant(MatrixAlgebra.minor(matrix, 0, c))
        return determinant

    @staticmethod
    def minor(matrix, row, col):
        # Return the minor of a matrix
        return Matrix([row[:col] + row[col+1:] for row in (matrix.data[:row] + matrix.data[row+1:])])

    @staticmethod
    def inverse(matrix):
        determinant = MatrixAlgebra.determinant(matrix)
        if determinant == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")

        # Special case for 2x2 matrix
        if len(matrix.data) == 2 and len(matrix.data[0]) == 2:
            return Matrix([[matrix.data[1][1], -1*matrix.data[0][1]], 
                           [-1*matrix.data[1][0], matrix.data[0][0]]]) * (1/determinant)

        # General case for larger matrices
        cofactors = []
        for r in range(len(matrix.data)):
            cofactorRow = []
            for c in range(len(matrix.data)):
                minor = MatrixAlgebra.minor(matrix, r, c)
                cofactorRow.append(((-1)**(r+c)) * MatrixAlgebra.determinant(minor))
            cofactors.append(cofactorRow)
        cofactors = Matrix(cofactors).transpose()
        return cofactors * (1/determinant)

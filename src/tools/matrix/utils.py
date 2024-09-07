from .matrix import Matrix
class MatrixUtils:
    @staticmethod
    def reshape(matrix, new_rows, new_cols):
        flat_data = [val for row in matrix.data for val in row]
        if len(flat_data) != new_rows * new_cols:
            raise ValueError("Cannot reshape matrix of size {}x{} to {}x{}".format(matrix.rows, matrix.cols, new_rows, new_cols))
        reshaped = [flat_data[i*new_cols:(i+1)*new_cols] for i in range(new_rows)]
        return Matrix(reshaped)

    @staticmethod
    def slice(matrix, row_start, row_end, col_start, col_end):
        return Matrix([row[col_start:col_end] for row in matrix.data[row_start:row_end]])

    @staticmethod
    def concatenate(matrix1, matrix2, axis=0):
        if axis == 0:  # Vertical concatenation
            if matrix1.cols != matrix2.cols:
                raise ValueError("Matrices must have the same number of columns for vertical concatenation.")
            return Matrix(matrix1.data + matrix2.data)
        elif axis == 1:  # Horizontal concatenation
            if matrix1.rows != matrix2.rows:
                raise ValueError("Matrices must have the same number of rows for horizontal concatenation.")
            return Matrix([matrix1.data[i] + matrix2.data[i] for i in range(matrix1.rows)])
        else:
            raise ValueError("Axis must be 0 (vertical) or 1 (horizontal).")

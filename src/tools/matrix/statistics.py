class MatrixStatistics:
    @staticmethod
    def mean(matrix):
        return sum([sum(row) for row in matrix.data]) / (matrix.rows * matrix.cols)

    @staticmethod
    def variance(matrix):
        mean = MatrixStatistics.mean(matrix)
        return sum([(val - mean) ** 2 for row in matrix.data for val in row]) / (matrix.rows * matrix.cols)

    @staticmethod
    def stddev(matrix):
        variance = MatrixStatistics.variance(matrix)
        return variance ** 0.5

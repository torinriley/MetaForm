class FixedPrecision:
    def __init__(self, precision):
        self.precision = precision

    def add(self, a, b):
        return round(a + b, self.precision)

    def subtract(self, a, b):
        return round(a - b, self.precision)

    def multiply(self, a, b):
        return round(a * b, self.precision)

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return round(a / b, self.precision)


from ....tools.matrix.matrix import Matrix
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None  # No-op by default

    def backward(self):
        if self.grad is None:
            self.grad = Matrix.ones_like(self.data)
        self._backward()

    def __add__(self, other):
        result = Tensor(self.data + other.data)
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + result.grad
            if other.requires_grad:
                other.grad = other.grad + result.grad
        result._backward = _backward
        return result


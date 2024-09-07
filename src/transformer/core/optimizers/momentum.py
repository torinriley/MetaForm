from ....tools.matrix.matrix import Matrix

class Momentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [Matrix.zeros_like(param.data) for param in params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                self.velocity[i] = self.momentum * self.velocity[i] + self.lr * param.grad
                param.data -= self.velocity[i]

    def zero_grad(self):
        for param in self.params:
            param.grad = Matrix.zeros_like(param.data)

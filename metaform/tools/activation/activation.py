from tools.matrix import Matrix
from tools.utils import exp  
class ReLU:
    def __init__(self):
        pass
    
    def forward(self, x):
        return Matrix([[max(0, value) for value in row] for row in x.data])

class Sigmoid:
    def __init__(self):
        pass
    
    def forward(self, x):
        return Matrix([[1 / (1 + exp(-value)) for value in row] for row in x.data])

class Tanh:
    def __init__(self):
        pass
    
    def forward(self, x):
        return Matrix([[(exp(value) - exp(-value)) / (exp(value) + exp(-value)) for value in row] for row in x.data])

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        return Matrix([[value if value > 0 else self.alpha * value for value in row] for row in x.data])

class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, x):
        return Matrix([[value if value > 0 else self.alpha * (exp(value) - 1) for value in row] for row in x.data])

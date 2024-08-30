class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)

    def forward(self, x):
        x = np.dot(x, self.W1)
        x = np.maximum(0, x)  # ReLU activation
        x = np.dot(x, self.W2)
        return x
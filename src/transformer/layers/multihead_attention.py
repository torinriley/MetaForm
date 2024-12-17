from ...tools.matrix.matrix import Matrix

class MultiHeadSelfAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.Wq = Matrix.random(d_model, d_model) 
        self.Wk = Matrix.random(d_model, d_model)
        self.Wv = Matrix.random(d_model, d_model)
        self.Wo = Matrix.random(d_model, d_model)

    def split_heads(self, x, batch_size):
        # reshape and transpose using custom Matrix operations
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose((0, 2, 1, 3))

    def scaled_dot_product_attention(self, q, k, v):
        # matrix multiplication using your custom library
        matmul_qk = Matrix.matmul(q, k.transpose(-2, -1))
        
        # scale the attention scores
        dk = k.shape[-1]
        scaled_attention_logits = matmul_qk / Matrix.sqrt(dk)

        # apply softmax to the attention scores
        attention_weights = Matrix.exp(scaled_attention_logits)
        attention_weights = attention_weights / Matrix.sum(attention_weights, axis=-1, keepdims=True)

        output = Matrix.matmul(attention_weights, v)
        return output

    def forward(self, x):
        batch_size = x.shape[0]

        # apply weight matrices to the input x
        q = Matrix.matmul(x, self.Wq)
        k = Matrix.matmul(x, self.Wk)
        v = Matrix.matmul(x, self.Wv)

        # split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled dot-product attention
        scaled_attention = self.scaled_dot_product_attention(q, k, v)
        scaled_attention = scaled_attention.transpose((0, 2, 1, 3))

        # concatenate the attention heads and apply the final linear transformation
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        output = Matrix.matmul(concat_attention, self.Wo)

        return output

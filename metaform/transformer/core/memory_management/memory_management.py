from ....tools.matrix.utils import MatrixUtils
from ....tools.matrix import Matrix
import math

def apply_mask(attention_scores, mask):
    """Apply the given mask to the attention scores."""
    masked_scores = []
    for i, row in enumerate(attention_scores.data):
        masked_row = []
        for j, score in enumerate(row):
            if mask.data[i][j] == 0:  # assuming the mask is 0 where attention should be blocked
                masked_row.append(float('-inf'))  # assign -inf to prevent attention
            else:
                masked_row.append(score)
        masked_scores.append(masked_row)
    return Matrix(masked_scores)


def memory_efficient_attention(Q, K, V, mask=None):
    # instead of computing the entire attention matrix at once, do it in chunks to save memory
    attention_chunks = []
    for i in range(Q.cols):  # process by columns or rows in chunks
        q_chunk = Q[:, i:i+1]
        attention_scores = q_chunk * K.transpose()  
        if mask is not None:
            attention_scores = apply_mask(attention_scores, mask)
        attention_weights = softmax(attention_scores)  # apply softmax to get weights
        context_chunk = attention_weights * V  # multiply by value matrix
        attention_chunks.append(context_chunk)

    # Concatenate chunks to form the final attention matrix
    return MatrixUtils.concatenate(*attention_chunks, axis=1)

def softmax(matrix):
    """ Simple softmax implementation for attention scores. """
    max_vals = [[max(row) for row in matrix.data]]
    exp_data = [[math.exp(val - max_vals[i]) for val in row] for i, row in enumerate(matrix.data)]
    sum_exp_data = [sum(row) for row in exp_data]
    return Matrix([[val / sum_exp_data[i] for val in row] for i, row in exp_data])

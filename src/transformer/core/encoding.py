from typing import Optional
from ...tools.matrix.matrix import Matrix

def get_positional_encoding(seq_len: int, d_model: int, batch_size: Optional[int] = None, mode: str = 'sin_cos') -> Matrix:
    """
    Generates positional encoding for a sequence of a given length and model dimension.

    Args:
        seq_len (int): Length of the sequence (number of positions).
        d_model (int): Dimension of the model (embedding size).
        batch_size (Optional[int]): If provided, returns a batch of positional encodings.
        mode (str): The mode of positional encoding. Options:
            - 'sin_cos': Default sine and cosine encoding.
            - 'linear': Linear positional encoding.
            - 'random': Random positional encoding.
    
    Returns:
        Matrix: Positional encoding matrix of shape (seq_len, d_model) or (batch_size, seq_len, d_model).
    """

    def sin_cos_encoding(pos: int, d_model: int) -> Matrix:
        encoding = Matrix.zeros(d_model)
        for i in range(0, d_model, 2):
            encoding[i] = Matrix.sin(pos / (10000 ** (2 * i / d_model)))
            encoding[i + 1] = Matrix.cos(pos / (10000 ** (2 * i / d_model)))
        return encoding

    def linear_encoding(pos: int, d_model: int) -> Matrix:
        return Matrix.linspace(0, 1, d_model) * pos

    def random_encoding(d_model: int) -> Matrix:
        return Matrix.random(d_model)

    # Choose encoding mode
    if mode == 'sin_cos':
        encoding_func = sin_cos_encoding
    elif mode == 'linear':
        encoding_func = linear_encoding
    elif mode == 'random':
        encoding_func = random_encoding
    else:
        raise ValueError("Invalid mode. Choose from 'sin_cos', 'linear', or 'random'.")

    # generate positional encodings
    positional_encoding = Matrix([encoding_func(pos, d_model) for pos in range(seq_len)])

    if batch_size:
        positional_encoding = positional_encoding.tile((batch_size, 1, 1))
    
    return positional_encoding

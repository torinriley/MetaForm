from .transformer_block import TransformerBlock

class TransformerModel:
    """
    TransformerModel is a deep neural network model based on the Transformer architecture.

    This model consists of multiple stacked Transformer blocks. The number of blocks
    (layers) can be customized by changing the `num_layers` parameter during initialization.

    Parameters:
    -----------
    embed_size : int
        The size of the input embeddings.
    num_heads : int
        The number of attention heads in the multi-head attention mechanism.
    ff_hidden_size : int
        The number of hidden units in the feedforward network.
    num_layers : int, optional
        The number of Transformer blocks to stack in the model (default is 80).
    dropout : float, optional
        Dropout rate applied after attention and feedforward layers (default is 0.1).

    Methods:
    --------
    forward(x):
        Performs the forward pass through the entire Transformer model.
    """

    def __init__(self, embed_size, num_heads, ff_hidden_size, num_layers=80, dropout=0.1):
        """
        Initializes the TransformerModel with the specified number of layers.

        Parameters:
        -----------
        embed_size : int
            The size of the input embeddings.
        num_heads : int
            The number of attention heads in the multi-head attention mechanism.
        ff_hidden_size : int
            The number of hidden units in the feedforward network.
        num_layers : int, optional
            The number of Transformer blocks to stack in the model (default is 80).
        dropout : float, optional
            Dropout rate applied after attention and feedforward layers (default is 0.1).
        """
        self.layers = [TransformerBlock(embed_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)]

    def forward(self, x):
        """
        Performs the forward pass through the entire Transformer model.

        Parameters:
        -----------
        x : Matrix
            The input matrix representing the embedded input sequence.

        Returns:
        --------
        out : Matrix
            The output matrix after passing through all Transformer blocks.
        """
        for layer in self.layers:
            x = layer.forward(x)  # Pass through each Transformer block
        return x  # Final output after all layers

from .transformer_block import TransformerBlock
from ...tools.matrix import Matrix

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
    use_positional_encoding : bool, optional
        Whether to include positional encoding (default is True).
    checkpointing : bool, optional
        Whether to use gradient checkpointing to save memory (default is False).

    Methods:
    --------
    forward(x):
        Performs the forward pass through the entire Transformer model.
    """

    def __init__(self, embed_size, num_heads, ff_hidden_size, num_layers=80, dropout=0.1, 
                 use_positional_encoding=True, checkpointing=False):
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
        use_positional_encoding : bool, optional
            Whether to include positional encoding (default is True).
        checkpointing : bool, optional
            Whether to use gradient checkpointing to save memory (default is False).
        """
        self.layers = [TransformerBlock(embed_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)]
        self.use_positional_encoding = use_positional_encoding
        self.checkpointing = checkpointing

    def forward(self, x, positional_encoding=None):
        """
        Performs the forward pass through the entire Transformer model.

        Parameters:
        -----------
        x : Matrix
            The input matrix representing the embedded input sequence.
        positional_encoding : Matrix, optional
            The positional encoding to be added to the input sequence.

        Returns:
        --------
        out : Matrix
            The output matrix after passing through all Transformer blocks.
        """
        if self.use_positional_encoding and positional_encoding is not None:
            x = x + positional_encoding  # Add positional encoding if required

        for layer in self.layers:
            if self.checkpointing:
                x = self.checkpoint_layer(layer, x)  # Use gradient checkpointing
            else:
                x = layer.forward(x)  # Pass through each Transformer block

        return x  # Final output after all layers

    def checkpoint_layer(self, layer, x):
        """
        Applies gradient checkpointing for memory efficiency.

        Parameters:
        -----------
        layer : TransformerBlock
            The Transformer block to apply checkpointing to.
        x : Matrix
            The input matrix to the layer.

        Returns:
        --------
        out : Matrix
            The output matrix after the forward pass with checkpointing.
        """
        # Here, you would implement the logic for gradient checkpointing
        # This is typically done by recomputing the forward pass during the backward pass.
        # For now, we'll simulate it by just calling the forward pass.
        return layer.forward(x)

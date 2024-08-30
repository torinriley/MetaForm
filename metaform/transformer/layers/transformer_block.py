from .multihead_attention import MultiHeadAttention
from .feedforward import FeedForward
from .normalization import LayerNormalization
from ...tools.matrix import Matrix
from ...tools.training.gradient_checkpointing import GradientCheckpointing

class TransformerBlock:
    """
    TransformerBlock is a single block within a Transformer model.

    It consists of a multi-head self-attention layer followed by a feedforward layer,
    each with layer normalization and optional dropout.

    Parameters:
    -----------
    embed_size : int
        The size of the input embeddings.
    num_heads : int
        The number of attention heads in the multi-head attention mechanism.
    ff_hidden_size : int
        The number of hidden units in the feedforward network.
    dropout : float, optional
        Dropout rate applied after attention and feedforward layers (default is 0.1).
    checkpointing : bool, optional
        Whether to use gradient checkpointing for the block (default is False).
    """

    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.1, checkpointing=False):
        """
        Initializes the TransformerBlock.

        Parameters:
        -----------
        embed_size : int
            The size of the input embeddings.
        num_heads : int
            The number of attention heads in the multi-head attention mechanism.
        ff_hidden_size : int
            The number of hidden units in the feedforward network.
        dropout : float, optional
            Dropout rate applied after attention and feedforward layers (default is 0.1).
        checkpointing : bool, optional
            Whether to use gradient checkpointing for the block (default is False).
        """
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.feed_forward = FeedForward(embed_size, ff_hidden_size, dropout)
        self.checkpointing = checkpointing

        if self.checkpointing:
            self.checkpointer = GradientCheckpointing(self)

    def forward(self, x):
        """
        Performs the forward pass through the Transformer block.

        Parameters:
        -----------
        x : Matrix
            The input matrix representing the embedded input sequence.

        Returns:
        --------
        out : Matrix
            The output matrix after passing through the Transformer block.
        """
        # Multi-Head Self-Attention and Residual Connection
        if self.checkpointing:
            attention, _ = self.checkpointer.checkpoint(self.attention.forward, x, x, x)
        else:
            attention = self.attention.forward(x, x, x)

        x = self.norm1.forward(x + attention)

        # Feedforward Network and Residual Connection
        if self.checkpointing:
            feedforward, _ = self.checkpointer.checkpoint(self.feed_forward.forward, x)
        else:
            feedforward = self.feed_forward.forward(x)

        out = self.norm2.forward(x + feedforward)

        return out

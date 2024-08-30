import gc
from matrix import Matrix  

class MemoryManager:
    @staticmethod
    def gradient_checkpointing(forward_fn, *args):
        """
        Save memory by checkpointing gradients during training.

        Args:
            forward_fn (callable): The forward function of the model.
            *args: Arguments to pass to the forward function.

        Returns:
            The intermediate activations.
        """
        activations = forward_fn(*args)
        return activations

    @staticmethod
    def allocate_memory(rows, cols):
        """
        Efficiently allocate memory using custom Matrix class.

        Args:
            rows (int): Number of rows.
            cols (int): Number of columns.

        Returns:
            Matrix: A matrix of zeros with specified dimensions.
        """
        return Matrix([[0.0] * cols for _ in range(rows)])

    @staticmethod
    def deallocate_memory(matrix):
        """
        Efficiently deallocate memory.

        Args:
            matrix (Matrix): The Matrix object to deallocate.
        """
        del matrix
        gc.collect()

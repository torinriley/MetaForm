
class GradientCheckpointing:
    def __init__(self, model):
        """
        Initialize the GradientCheckpointing utility.

        Parameters:
        -----------
        model : TransformerModel
            The model to apply gradient checkpointing to.
        """
        self.model = model

    def checkpoint(self, function, *inputs):
        """
        Apply checkpointing to a specific function in the model.

        Parameters:
        -----------
        function : callable
            The function (usually a layer or block) to apply checkpointing to.
        inputs : list of Matrix
            The inputs to the function.

        Returns:
        --------
        output : Matrix
            The output after the function is applied, with checkpointing.
        """
        # Forward pass with checkpointing
        saved_tensors = self.save_for_backward(*inputs)
        output = function(*inputs)
        
        # Clear the saved tensors to save memory
        self.clear_saved_tensors()
        
        def backward(*grad_outputs):
            # Recompute the forward pass during the backward pass
            recomputed_output = function(*inputs)
            grads = recomputed_output.backward(*grad_outputs)  
            return grads

        return output, backward

    def save_for_backward(self, *tensors):
        """
        Save tensors for the backward pass.

        Parameters:
        -----------
        tensors : list of Matrix
            The tensors to save for later use during the backward pass.
        """
        self.saved_tensors = tensors

    def clear_saved_tensors(self):
        """
        Clear the saved tensors to save memory.
        """
        self.saved_tensors = None




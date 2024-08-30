from tools.training import MixedPrecision

class PrecisionOptimizer:
    def __init__(self, scale_factor=2**16, dynamic_loss_scaling=True, initial_scale=2.**15):
        """
        Initialize the PrecisionOptimizer for mixed precision training.

        Parameters:
        -----------
        scale_factor : float, optional
            The factor by which to scale gradients to prevent underflow.
        dynamic_loss_scaling : bool, optional
            Whether to use dynamic loss scaling to prevent underflow.
        initial_scale : float, optional
            The initial scale for dynamic loss scaling.
        """
        self.mixed_precision = MixedPrecision()
        self.scale_factor = scale_factor
        self.dynamic_loss_scaling = dynamic_loss_scaling
        self.loss_scale = initial_scale
        self.min_scale = 1.0
        self.scale_adjustment_factor = 2.0

    def scale_gradients(self, gradients):
        """
        Scale gradients to float16 precision.
        """
        if self.dynamic_loss_scaling:
            return self.mixed_precision.scale_gradients(gradients, self.loss_scale)
        return self.mixed_precision.scale_gradients(gradients, self.scale_factor)

    def unscale_gradients(self, gradients):
        """
        Unscale gradients back to float32 precision.
        """
        if self.dynamic_loss_scaling:
            return self.mixed_precision.unscale_gradients(gradients, self.loss_scale)
        return self.mixed_precision.unscale_gradients(gradients, self.scale_factor)

    def update_loss_scale(self, overflow_occurred):
        """
        Adjust the loss scale dynamically based on whether an overflow occurred.

        Parameters:
        -----------
        overflow_occurred : bool
            Whether an overflow occurred during the last operation.
        """
        if overflow_occurred:
            self.loss_scale = max(self.loss_scale / self.scale_adjustment_factor, self.min_scale)
        else:
            self.loss_scale *= self.scale_adjustment_factor

    def clip_gradients(self, gradients, clip_value):
        """
        Clip gradients to prevent exploding gradients.

        Parameters:
        -----------
        gradients : list of Matrix
            The list of gradients to clip.
        clip_value : float
            The maximum absolute value for gradient clipping.

        Returns:
        --------
        clipped_gradients : list of Matrix
            The clipped gradients.
        """
        return [self._clip(grad, clip_value) for grad in gradients]
    
    def _clip(self, tensor, clip_value):
        """
        Clip tensor values to within the specified range.

        Parameters:
        -----------
        tensor : Matrix
            The tensor to clip.
        clip_value : float
            The maximum absolute value to clip to.

        Returns:
        --------
        clipped_tensor : Matrix
            The clipped tensor.
        """
        clipped_data = [
            [max(min(value, clip_value), -clip_value) for value in row]
            for row in tensor.data
        ]
        return tensor.__class__(clipped_data)  # Assuming the tensor has a data attribute and a constructor that accepts data

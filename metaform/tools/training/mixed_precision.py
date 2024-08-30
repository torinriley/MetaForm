class MixedPrecision:
    def __init__(self):
        pass

    def to_float16(self, tensor):
        return tensor.astype('float16')
    
    def to_float32(self, tensor):
        return tensor.astype('float32')
    
    def scale_gradients(self, gradients, scale):
        return [grad * scale for grad in gradients]

    def unscale_gradients(self, gradients, scale):
        return [grad / scale for grad in gradients]

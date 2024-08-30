from multiprocessing import Manager

class Parallelization:
    @staticmethod
    def distribute_layers(layers, devices):
        """
        Distribute model layers across multiple devices.

        Parameters:
        -----------
        layers : list
            A list of model layers to distribute.
        devices : list
            A list of devices (e.g., 'cpu', 'cuda:0', 'cuda:1') to distribute layers across.

        Returns:
        --------
        layer_dict : dict
            A dictionary mapping each device to its assigned layer(s).
        """
        manager = Manager()
        layer_dict = manager.dict()
        
        for i, layer in enumerate(layers):
            device = devices[i % len(devices)] 
            if device not in layer_dict:
                layer_dict[device] = []
            layer_dict[device].append(layer)

        return dict(layer_dict) 

    @staticmethod
    def aggregate_gradients(gradients_list):
        """
        Aggregate gradients from different devices.

        Parameters:
        -----------
        gradients_list : list of list of matrices
            A list containing gradient matrices from each device.

        Returns:
        --------
        aggregated_gradients : list of matrices
            A list of aggregated gradients.
        """
        aggregated_gradients = []
        
        for grads in zip(*gradients_list):
            aggregated = sum(grads)
            aggregated_gradients.append(aggregated)
        
        return aggregated_gradients

    @staticmethod
    def update_parameters(model, aggregated_gradients, learning_rate=0.01):
        """
        Update model parameters based on aggregated gradients.

        Parameters:
        -----------
        model : Model
            The model whose parameters need to be updated.
        aggregated_gradients : list of matrices
            A list of aggregated gradients to update the model parameters.
        learning_rate : float, optional
            The learning rate to apply during the parameter update (default is 0.01).
        """
        for param, grad in zip(model.parameters(), aggregated_gradients):
            param -= learning_rate * grad 

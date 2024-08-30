# tools/training/distributed_training.py

import socket
import threading
import pickle
from ...tools.matrix.matrix import Matrix

class DistributedTraining:
    def __init__(self, model, num_devices=1, backend='socket', host='localhost', port=12345):
        """
        Initialize distributed training.

        Parameters:
        -----------
        model : TransformerModel
            The model to be trained in a distributed manner.
        num_devices : int, optional
            The number of devices to distribute the training across (default is 1).
        backend : str, optional
            The backend to use for communication (default is 'socket').
        host : str, optional
            The hostname for the socket communication (default is 'localhost').
        port : int, optional
            The port number for socket communication (default is 12345).
        """
        self.model = model
        self.num_devices = num_devices
        self.backend = backend
        self.host = host
        self.port = port

        if self.backend == 'socket':
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(num_devices)

            self.clients = []
            for _ in range(num_devices):
                client_socket, _ = self.server_socket.accept()
                self.clients.append(client_socket)
    
    def distribute_data(self, data):
        """
        Split data across devices.

        Parameters:
        -----------
        data : list of Matrix
            The data to be split and distributed.

        Returns:
        --------
        List of data chunks for each device.
        """
        chunk_size = len(data) // self.num_devices
        return [data[i * chunk_size:(i + 1) * chunk_size] for i in range(self.num_devices)]

    def send_to_devices(self, data_chunks):
        """
        Send data chunks to devices.

        Parameters:
        -----------
        data_chunks : list of Matrix
            The data chunks to be sent to each device.
        """
        for i, chunk in enumerate(data_chunks):
            self._send_data(self.clients[i], chunk)

    def _send_data(self, client_socket, data):
        """
        Internal method to send data to a client.

        Parameters:
        -----------
        client_socket : socket
            The socket corresponding to the client.
        data : Matrix
            The data to be sent.
        """
        serialized_data = pickle.dumps(data)
        client_socket.sendall(serialized_data)

    def _receive_data(self, client_socket):
        """
        Internal method to receive data from a client.

        Parameters:
        -----------
        client_socket : socket
            The socket corresponding to the client.

        Returns:
        --------
        The received data.
        """
        serialized_data = client_socket.recv(4096)  # You might need a loop to receive all data
        return pickle.loads(serialized_data)

    def synchronize_gradients(self, gradients):
        """
        Synchronize gradients across devices.

        Parameters:
        -----------
        gradients : list of Matrix
            The gradients from each device.

        Returns:
        --------
        Averaged gradients across all devices.
        """
        # Sum gradients across all devices
        total_gradients = [Matrix.zeros_like(g) for g in gradients[0]]
        for g_list in gradients:
            for i, g in enumerate(g_list):
                total_gradients[i] += g

        # Average the gradients
        averaged_gradients = [g / self.num_devices for g in total_gradients]

        # Broadcast the averaged gradients back to each device
        for client in self.clients:
            self._send_data(client, averaged_gradients)

        return averaged_gradients

    def update_model(self, averaged_gradients):
        """
        Update the model weights using the averaged gradients.

        Parameters:
        -----------
        averaged_gradients : list of Matrix
            The averaged gradients to apply to the model.
        """
        self.model.update_weights(averaged_gradients)

    def train(self, data_loader, optimizer, num_epochs=1):
        """
        Train the model in a distributed manner.

        Parameters:
        -----------
        data_loader : DataLoader
            The data loader for loading batches of data.
        optimizer : PrecisionOptimizer
            The optimizer to use for training.
        num_epochs : int, optional
            The number of epochs to train for (default is 1).
        """
        for epoch in range(num_epochs):
            for batch in data_loader:
                # Split and distribute the data
                data_chunks = self.distribute_data(batch)
                self.send_to_devices(data_chunks)

                # Receive gradients from each device
                gradients = []
                for client in self.clients:
                    gradients.append(self._receive_data(client))

                # Synchronize and update gradients
                averaged_gradients = self.synchronize_gradients(gradients)
                optimizer.update_model(averaged_gradients)

            print(f"Epoch {epoch + 1}/{num_epochs} completed.")


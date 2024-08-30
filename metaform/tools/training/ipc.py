from multiprocessing import Pipe, Process, Lock, Manager
import threading

class IPC:
    def __init__(self):
        # Create a pipe for communication
        self.parent_conn, self.child_conn = Pipe()
        # A lock for thread-safe operations
        self.lock = Lock()

    def send(self, message):
        with self.lock:
            self.parent_conn.send(message)

    def receive(self):
        with self.lock:
            return self.child_conn.recv()

    def close(self):
        self.parent_conn.close()
        self.child_conn.close()

    def synchronize(self, data):
        """
        Synchronize model parameters or gradients.
        """
        # Send data to other processes
        self.send(data)
        # Wait for acknowledgment (or synchronization from other processes)
        return self.receive()

# Example usage for distributed training
def worker(ipc):
    # Example: Worker sending and receiving model parameters
    ipc.send("Model parameters")
    print("Received:", ipc.receive())

if __name__ == "__main__":
    ipc = IPC()
    p1 = Process(target=worker, args=(ipc,))
    p2 = Process(target=worker, args=(ipc,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

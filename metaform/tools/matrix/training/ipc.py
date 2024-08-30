from multiprocessing import Pipe, Process

class IPC:
    def __init__(self):
        self.parent_conn, self.child_conn = Pipe()

    def send(self, message):
        self.parent_conn.send(message)

    def receive(self):
        return self.child_conn.recv()

    def close(self):
        self.parent_conn.close()
        self.child_conn.close()

# Example usage
def sender(ipc):
    ipc.send("Hello from sender")

def receiver(ipc):
    print("Received:", ipc.receive())

if __name__ == "__main__":
    ipc = IPC()
    p1 = Process(target=sender, args=(ipc,))
    p2 = Process(target=receiver, args=(ipc,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

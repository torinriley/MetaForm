class MemoryManager:
    def __init__(self):
        self.memory = {}

    def allocate(self, name, size):
        """Allocate memory with a specific size."""
        if name in self.memory:
            raise ValueError(f"Memory already allocated for {name}")
        self.memory[name] = [None] * size

    def deallocate(self, name):
        """Deallocate memory."""
        if name in self.memory:
            del self.memory[name]
        else:
            raise ValueError(f"No memory allocated for {name}")

    def write(self, name, index, value):
        """Write a value to a specific memory location."""
        if name in self.memory:
            if index >= len(self.memory[name]):
                raise IndexError("Index out of bounds")
            self.memory[name][index] = value
        else:
            raise ValueError(f"No memory allocated for {name}")

    def read(self, name, index):
        """Read a value from a specific memory location."""
        if name in self.memory:
            if index >= len(self.memory[name]):
                raise IndexError("Index out of bounds")
            return self.memory[name][index]
        else:
            raise ValueError(f"No memory allocated for {name}")

#
# memory_pool.pyx

cdef extern from "memory_pool.h":
    cdef cppclass MemoryPool:
        MemoryPool(size_t blockSize, size_t poolSize)
        vector[char] allocate()
        void deallocate(vector[char] block)

cdef class PyMemoryPool:
    cdef MemoryPool* pool

    def __cinit__(self, size_t blockSize, size_t poolSize):
        self.pool = new MemoryPool(blockSize, poolSize)

    def __dealloc__(self):
        del self.pool

    def allocate(self):
        block = self.pool.allocate()
        return bytes(block)

    def deallocate(self, block: bytes):
        self.pool.deallocate(block)
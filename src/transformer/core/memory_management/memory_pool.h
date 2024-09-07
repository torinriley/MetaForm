#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <vector>
#include <queue>

class MemoryPool {
public:
    MemoryPool(size_t blockSize, size_t poolSize);

    std::vector<char> allocate();      // Allocates a block of memory
    void deallocate(std::vector<char> block);  // Deallocates (returns) the block to the pool

private:
    size_t blockSize;
    std::queue<std::vector<char>> pool;  // The pool of available blocks
};

#endif
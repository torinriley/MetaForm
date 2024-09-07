#include "memory_pool.h"
#include <vector>
#include <queue>

MemoryPool::MemoryPool(size_t blockSize, size_t poolSize) : blockSize(blockSize) {
    for (size_t i = 0; i < poolSize; ++i) {
        pool.push(std::vector<char>(blockSize));
    }
}

std::vector<char> MemoryPool::allocate() {
    if (pool.empty()) {
        return std::vector<char>(blockSize);  // Allocate new if pool is empty
    }
    std::vector<char> block = pool.front();
    pool.pop();
    return block;
}

void MemoryPool::deallocate(std::vector<char> block) {
    pool.push(block);
}
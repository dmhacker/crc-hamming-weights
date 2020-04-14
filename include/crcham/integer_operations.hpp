#ifndef CRCHAM_INTEGER_OPERATIONS
#define CRCHAM_INTEGER_OPERATIONS

#include <cstddef>
#include <cstdint>

namespace crcham {
    __device__ __host__
    size_t ctz64(uint64_t x);

    __device__ __host__
    uint64_t ncr64(uint64_t n, uint64_t k);
}

#endif

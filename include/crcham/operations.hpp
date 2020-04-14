#ifndef CRCHAM_OPERATIONS_HPP
#define CRCHAM_OPERATIONS_HPP

#include <cstddef>
#include <cstdint>

namespace crcham {
    __device__ __host__
    uint64_t ncr64(uint64_t n, uint64_t k);
}

#endif

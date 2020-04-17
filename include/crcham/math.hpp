#ifndef CRCHAM_MATH_HPP 
#define CRCHAM_MATH_HPP 

#include <cstddef>
#include <cstdint>

namespace crcham {
    __device__ __host__
    uint64_t ncrll(uint64_t n, uint64_t k);
}

#endif

#ifndef CRCHAM_PERMUTE_HPP 
#define CRCHAM_PERMUTE_HPP

#include <cstddef>
#include <cstdint>

namespace crcham {

__device__ __host__
void permute(uint32_t* arr, size_t len, uint64_t n, size_t m, size_t k);

__device__ __host__
size_t popcount(uint32_t* arr, size_t len);

}

#endif

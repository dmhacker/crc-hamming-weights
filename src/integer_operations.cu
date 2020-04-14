#include <crcham/integer_operations.hpp>

namespace crcham {

__device__ __host__
size_t ctz64(uint64_t x) 
{
    // Assume x is not 0, otherwise this is undefined
    size_t result = 1;
    if ((x & 0xFFFFFFFF) == 0) {
        result += 32; 
        x = x >> 32;
    }
    if ((x & 0x0000FFFF) == 0) {
        result += 16; 
        x = x >>16;
    }
    if ((x & 0x000000FF) == 0) {
        result += 8; 
        x = x >> 8;
    }
    if ((x & 0x0000000F) == 0) {
        result += 4; 
        x = x >> 4;
    }
    if ((x & 0x00000003) == 0) {
        result += 2;
        x = x >> 2;
    }
    return result - (x & 1);
}

__device__ __host__
uint64_t ncr64(uint64_t n, uint64_t k)
{
    // See https://stackoverflow.com/questions/9330915/number-of-combinations-n-choose-r-in-c
    if (k > n) {
        return 0;
    }
    if (k * 2 > n) {
        k = n - k;
    }
    if (k == 0) {
        return 1;
    }
    uint64_t result = n;
    for( uint64_t i = 2; i <= k; ++i ) {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

}

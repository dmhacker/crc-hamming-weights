#include <crcham/operations.hpp>

namespace crcham {

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

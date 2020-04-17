#include <crcham/math.hpp>
#include <crcham/permute.hpp>

namespace crcham {

__device__ __host__
void permute(uint32_t* arr, size_t len, uint64_t n, size_t m, size_t k) {
    memset(arr, 0, len * sizeof(uint32_t));
    for (size_t i = 0; i < m; i++) {
        uint64_t binom = ncrll(m - i - 1, k); 
        if (n >= binom) {
            // Align last index with the end of the buffer
            size_t ia = 32 * len - m + i;
            arr[ia / 32] |= (1 << (31 - ia % 32));
            n -= binom;
            k--;
        }
    }
}

__device__ __host__
size_t popcount(uint32_t* arr, size_t len) {
    size_t ones = 0;
    for (size_t i = 0; i < len; i++) {
#ifdef __CUDA_ARCH__
        ones += __popc(arr[i]);
#else
        ones += __builtin_popcount(arr[i]);
#endif
    }
    return ones;
}

}

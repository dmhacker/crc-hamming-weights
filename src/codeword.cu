#include <crcham/codeword.hpp>
#include <crcham/math.hpp>

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
uint64_t extract(uint8_t* arr, size_t len, size_t bits, size_t polybits) {
    uint64_t shiftr = 0;
    for (size_t i = 0; i < polybits; i++) {
        size_t ia = 8 * len - bits + i;
        size_t byte = ia / 8;
        uint8_t selector = 1 << (7 - ia % 8);
        if (arr[byte] & selector) {
            shiftr |= 1;
            arr[byte] ^= selector;
        }
        shiftr <<= 1;
    }
    return shiftr;
}

__device__ __host__
size_t popcount(const uint32_t* arr, size_t len) {
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

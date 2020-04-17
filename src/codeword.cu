#include <crcham/codeword.hpp>
#include <crcham/math.hpp>

#include <cassert>
#include <cstdio>

namespace crcham {

__device__ __host__
void permute(uint8_t* arr, size_t len, uint64_t n, size_t m, size_t k) {
    memset(arr, 0, len * sizeof(uint8_t));
    uint64_t binom = ncrll(m - 1, k); 
    for (size_t i = 0; i < m; i++) {
        size_t j = m - i - 1;
        if (n >= binom) {
            // Align last index with the end of the buffer
            size_t ia = 8 * len - m + i;
            arr[ia / 8] |= (1 << (7 - ia % 8));
            n -= binom;
            binom *= k;
            k--;
        }
        else {
            binom *= (j - k);
        }
        binom /= ((j > 0) * (j - 1) + 1);
    }
}

__device__ __host__
uint64_t extract(uint8_t* arr, size_t len, size_t bits, size_t polybits) {
    uint64_t shiftr = 0;
    for (size_t i = 0; i < polybits; i++) {
        size_t ia = 8 * len - bits + i;
        size_t byte = ia / 8;
        uint8_t selector = 1 << (7 - ia % 8);
        shiftr <<= 1;
        if (arr[byte] & selector) {
            shiftr |= 1;
            arr[byte] ^= selector;
        }
    }
    return shiftr;
}

__device__ __host__
size_t popcount(const uint8_t* arr, size_t len) {
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

#ifndef CRCHAM_CODEWORD_HPP
#define CRCHAM_CODEWORD_HPP

#include <cstddef>
#include <cstdint>
#include <stdio.h>

#include <crcham/operations.hpp>

namespace crcham {

template <size_t N>
class Codeword {
    uint32_t d_arr_p[N];

public:
    __device__ __host__
    Codeword();

    __device__ __host__
    void permute(uint64_t n, size_t m, size_t k);

    __device__ __host__
    uint32_t* get();
    __device__ __host__
    size_t popcount() const;

    __device__ __host__
    bool operator==(const Codeword&) const;
    __device__ __host__
    bool operator!=(const Codeword&) const;
};

template <size_t N>
__device__ __host__
Codeword<N>::Codeword()
{
    memset(d_arr_p, 0, N * sizeof(uint32_t));
}

template <size_t N>
__device__ __host__
uint32_t* Codeword<N>::get()
{
    return d_arr_p;
}

template <size_t N>
__device__ __host__
size_t Codeword<N>::popcount() const
{
    size_t ones = 0;
    for (size_t i = 0; i < N; i++) {
#ifdef __CUDA_ARCH__
        ones += __popc(d_arr_p[i]);
#else
        ones += __builtin_popcount(d_arr_p[i]);
#endif
    }
    return ones;
}

template <size_t N>
__device__ __host__
void Codeword<N>::permute(uint64_t n, size_t m, size_t k) {
    memset(d_arr_p, 0, N * sizeof(uint32_t));
    for (size_t i = 0; i < m; i++) {
        uint64_t binom = ncr64(m - i - 1, k); 
        if (n >= binom) {
            // ai = absolute index, aligns indices with 
            // the right edge of the available array
            size_t ai = 32 * N - m + i;
            d_arr_p[ai / 32] |= (1 << (31 - ai % 32));
            n -= binom;
            k--;
        }
    }
}

template <size_t N>
__device__ __host__
bool Codeword<N>::operator==(const Codeword& other) const {
    for (size_t i = 0; i < N; i++) {
        if (d_arr_p[i] != other.d_arr_p[i]) {
            return false;
        }
    }
    return true;
}

template <size_t N>
__device__ __host__
bool Codeword<N>::operator!=(const Codeword& other) const {
    return !(*this == other);
}

}

#endif

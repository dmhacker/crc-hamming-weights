#ifndef CRCHAM_CODEWORD
#define CRCHAM_CODEWORD

#include <cstddef>
#include <cstdint>

#include <crcham/operations.hpp>

namespace crcham {

template <size_t N>
class Codeword {
    uint64_t d_arr_p[N];

public:
    __device__ __host__
    Codeword();

    __device__ __host__
    void permute(uint64_t n, size_t m, size_t k);

    __device__ __host__
    uint64_t* get();
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
    memset(d_arr_p, 0, N * sizeof(uint64_t));
}

template <size_t N>
__device__ __host__
uint64_t* Codeword<N>::get()
{
    return d_arr_p;
}

template <size_t N>
__device__ __host__
size_t Codeword<N>::popcount() const
{
    size_t ones = 0;
    for (size_t i = 0; i < N; i++) {
        ones += __popcll(d_arr_p[i]);
    }
    return ones;
}

template <size_t N>
__device__ __host__
void Codeword<N>::permute(uint64_t n, size_t m, size_t k) {
    memset(d_arr_p, 0, N * sizeof(uint64_t));
    for (size_t i = 0; i < m; i++) {
        uint64_t binom = ncr64(m - i - 1, k); 
        size_t offset = i / 64;
        d_arr_p[offset] <<= 1;
        if (n >= binom) {
            d_arr_p[offset] |= 1;
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

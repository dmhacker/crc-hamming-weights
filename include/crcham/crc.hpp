#ifndef CRCHAM_CRC_HPP
#define CRCHAM_CRC_HPP

#include <cstddef>
#include <cstdint>

#include <crcham/codeword.hpp>

namespace crcham {

class NaiveCRC {
private:
    size_t d_length;
    uint64_t d_generator;

public:
    __device__ __host__
    NaiveCRC(uint64_t koopman);

    __device__ __host__
    uint64_t polynomial() const;
    __device__ __host__
    size_t length() const;

    template<size_t N>
    __device__  __host__
    bool verify(Codeword<N> codeword);
};


class TabularCRC {
private:
    size_t d_length;
    uint64_t d_generator;
    uint64_t d_table[256];

public:
    __device__  __host__
    TabularCRC(uint64_t koopman);

    __device__ __host__
    uint64_t polynomial() const;
    __device__ __host__
    size_t length() const;

    template<size_t N>
    __device__  __host__
    bool verify(Codeword<N> codeword);
};

template<size_t N>
__device__ __host__
bool NaiveCRC::verify(Codeword<N> codeword) {
    // TODO: http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html
    return false;
}

template<size_t N>
__device__ __host__
bool TabularCRC::verify(Codeword<N> codeword) {
    // TODO: http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html
    return false;
}

}

#endif

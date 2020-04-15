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
    __device__ 
    NaiveCRC(uint64_t koopman);

    template<size_t N>
    __device__ 
    bool verify(Codeword<N> codeword);
};


class TabularCRC {
private:
    size_t d_length;
    uint64_t d_generator;
    uint64_t d_table[256];

public:
    __device__ 
    TabularCRC(uint64_t koopman);

    template<size_t N>
    __device__ 
    bool verify(Codeword<N> codeword);
};

template<size_t N>
__device__ 
bool NaiveCRC::verify(Codeword<N> codeword) {
    // TODO: http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html
    return false;
}

template<size_t N>
__device__ 
bool TabularCRC::verify(Codeword<N> codeword) {
    // TODO: http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html
    return false;
}

}

#endif

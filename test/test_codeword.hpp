#ifndef TEST_CODEWORD_HPP
#define TEST_CODEWORD_HPP

#include <cassert>
#include <stdio.h>

#include <crcham/codeword.hpp>

template <size_t N>
__device__ __host__
void testCodewordEqual() {
    crcham::Codeword<N> cw1;
    crcham::Codeword<N> cw2;
    assert(cw1 == cw2);
    cw1.get()[N - 1] = 0x1;
    cw2.get()[N - 1] = 0x1;
    assert(cw1 == cw2);
}

template <size_t N>
__device__ __host__
void testCodewordInequal() {
    crcham::Codeword<N> cw1;
    crcham::Codeword<N> cw2;
    cw1.get()[N - 1] = 0xF;
    cw2.get()[N - 1] = 0xA;
    assert(cw1 != cw2);
}

template <size_t N>
__device__ __host__
void testCodewordPermute(size_t bitcount, size_t popcount) {
    crcham::Codeword<N> cw1;
    for (size_t i = 0; i < 8; i++) {
        cw1.permute(i, bitcount, popcount);
        assert(cw1.popcount() == popcount);
    }
}

#endif

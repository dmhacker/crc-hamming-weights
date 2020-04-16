#include "catch.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>

#include <crcham/codeword.hpp>

namespace {

template <size_t N>
void testCodewordEqual() {
    crcham::Codeword<N> cw1;
    crcham::Codeword<N> cw2;
    assert(cw1 == cw2);
    cw1.get()[N - 1] = 0x1;
    cw2.get()[N - 1] = 0x1;
    assert(cw1 == cw2);
}

template <size_t N>
void testCodewordInequal() {
    crcham::Codeword<N> cw1;
    crcham::Codeword<N> cw2;
    cw1.get()[N - 1] = 0xF;
    cw2.get()[N - 1] = 0xA;
    assert(cw1 != cw2);
}

template <size_t N>
void testCodewordPermute(size_t bitcount, size_t popcount, size_t iterations) {
    crcham::Codeword<N> cw1;
    for (size_t i = 0; i < iterations; i++) {
        cw1.permute(i, bitcount, popcount);
        assert(cw1.popcount() == popcount);
    }
}

}

TEST_CASE("Codeword equality comparisons") {
    testCodewordEqual<1>();
    testCodewordEqual<57>();
    testCodewordEqual<271>();
    testCodewordInequal<1>();
    testCodewordInequal<57>();
    testCodewordInequal<271>();
}

TEST_CASE("Codeword permutation generation") {
    for (size_t m = 64; m <= 256; m++) {
        for (size_t k = 1; k < 8; k++) {
            // Parameters chosen such that that ((k - 1) choose w) doesn't 
            // exceed the memory limits of a 64-bit unsigned integer
            testCodewordPermute<8>(m, k, 16);
        }
    }   
}

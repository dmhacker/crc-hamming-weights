#include "catch.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>

#include <crcham/permute.hpp>

namespace {

void testBufferPermute(size_t bitcount, size_t popcount, size_t iterations) {
    size_t buflen = bitcount % 8 == 0 ? bitcount / 8 : bitcount / 8 + 1;
    uint32_t* buffer = new uint32_t[buflen];
    for (size_t i = 0; i < iterations; i++) {
        crcham::permute(buffer, buflen, i, bitcount, popcount);
        assert(crcham::popcount(buffer, buflen) == popcount);
    }
}

}

TEST_CASE("Buffer permutation generation") {
    for (size_t m = 64; m <= 256; m++) {
        for (size_t k = 1; k < 8; k++) {
            // Parameters chosen such that that ((k - 1) choose w) doesn't 
            // exceed the memory limits of a 64-bit unsigned integer
            testBufferPermute(m, k, 16);
        }
    }   
}

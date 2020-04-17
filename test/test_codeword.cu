#include "catch.hpp"

#include <cstdint>
#include <cstdio>

#include <crcham/codeword.hpp>

namespace {

void testPermute(size_t bitcount, size_t popcount, size_t iterations) {
    size_t buflen = bitcount % 8 == 0 ? bitcount / 8 : bitcount / 8 + 1;
    uint8_t* buffer = new uint8_t[buflen];
    for (size_t i = 0; i < iterations; i++) {
        crcham::permute(buffer, buflen, i, bitcount, popcount);
        REQUIRE(crcham::popcount(buffer, buflen) == popcount);
    }
}

void testExtract(size_t bitcount, size_t extractcount) {
    size_t buflen = bitcount % 8 == 0 ? bitcount / 8 : bitcount / 8 + 1;
    uint8_t* buffer = new uint8_t[buflen];
    for (size_t i = 0; i < buflen; i++) {
        buffer[i] = 255;
    }
    uint64_t result = crcham::extract(buffer, buflen, bitcount, extractcount);
    size_t bufpop = crcham::popcount(buffer, buflen);
    size_t respop = __builtin_popcountll(result);
    REQUIRE(respop == extractcount);
    REQUIRE(respop + bufpop == buflen * 8);
}

}

TEST_CASE("Codeword permutation generation") {
    for (size_t m = 64; m <= 256; m++) {
        for (size_t k = 1; k < 8; k++) {
            // Parameters chosen such that that ((k - 1) choose w) doesn't 
            // exceed the memory limits of a 64-bit unsigned integer
            testPermute(m, k, 16);
        }
    }   
}

TEST_CASE("CRC extraction from codeword") {
    for (size_t m = 64; m <= 256; m++) {
        for (size_t k = 1; k <= 64; k++) {
            testExtract(m, k);
        }
    }
}

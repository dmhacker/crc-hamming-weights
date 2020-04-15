#ifndef TEST_CRC_HPP
#define TEST_CRC_HPP

#include <cassert>
#include <cstring>
#include <iostream>
#include <stdio.h>

#include <crcham/crc.hpp>

__device__ __host__ size_t strsize(const char * message) 
{
    size_t sz = 0;
    while (true) {
        if (message[sz] == '\0') {
            return sz;
        }
        sz++;
    }
    return 0;
}

__device__ __host__ void testCRCMetadata(uint64_t koopman, uint64_t normal, size_t bits)
{
    crcham::NaiveCRC ncrc(koopman);
    assert(normal == ncrc.polynomial());
    assert(bits == ncrc.length());
    crcham::TabularCRC tcrc(koopman);
    assert(normal == tcrc.polynomial());
    assert(bits == tcrc.length());
}

__device__ __host__ void testCRCComputeTabular(const char message[], uint64_t koopman, uint64_t crc)
{
    crcham::TabularCRC tcrc(koopman);
    uint64_t result = tcrc.compute(reinterpret_cast<const uint8_t*>(message), strsize(message));
    assert(crc == result);
}

#endif

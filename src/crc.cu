#include <crcham/crc.hpp>

#include <cassert>

namespace crcham {

__device__ __host__
NaiveCRC::NaiveCRC(uint64_t koopman)
    : d_generator(koopman)
{
#ifdef __CUDA_ARCH__
    d_polylen = 64 - __clzll(koopman);
#else
    d_polylen = 64 - __builtin_clzll(koopman);
#endif
    if (d_polylen > 0) {
        d_generator ^= 1ULL << (d_polylen - 1);
    }
    d_generator <<= 1;
    d_generator |= 1;
}

__device__ __host__
uint64_t NaiveCRC::polynomial() const {
    return d_generator;
}

__device__ __host__
size_t NaiveCRC::length() const {
    return d_polylen;
}

__device__ __host__
uint64_t NaiveCRC::compute(const uint8_t* bytes, size_t bytelen) const {
    // NOTE: this is essentially a replication of a shift register
    uint64_t shiftr = 0;
    const size_t bitlen = bytelen * 8;
    for (size_t idx = 0; idx < bitlen + d_polylen; idx++) {
        auto bit = (idx < bitlen) ? 
            ((bytes[idx / 8] >> (7 - (idx % 8))) & 1) : 0;
        auto msb = (shiftr >> (d_polylen - 1)) & 1;
        shiftr <<= 1; 
        shiftr |= bit;
        shiftr &= (1ULL << d_polylen) - 1;
        shiftr ^= msb ? d_generator : 0;
    }
    return shiftr;
}

 __device__ __host__
TabularCRC::TabularCRC(uint64_t koopman) 
    : d_generator(koopman)
{
#ifdef __CUDA_ARCH__
    d_polylen = 64 - __clzll(koopman);
#else
    d_polylen = 64 - __builtin_clzll(koopman);
#endif
    assert(d_polylen >= 8);
    uint64_t mask = 1ULL << (d_polylen - 1);
    d_generator ^= mask;
    d_generator <<= 1;
    d_generator |= 1;
    for (uint64_t byte = 0; byte < 256; byte++) {
        uint64_t result = byte << (d_polylen - 8);
        for (size_t b = 0; b < 8; b++) {
            if (result & mask) {
                result <<= 1;
                result ^= d_generator;
            }
            else {
                result <<= 1;
            }
            result &= (1ULL << d_polylen) - 1;
        }
        d_table[byte] = result;
    }
}

__device__ __host__
uint64_t TabularCRC::polynomial() const {
    return d_generator;
}

__device__ __host__
size_t TabularCRC::length() const {
    return d_polylen;
}

__device__ __host__
uint64_t TabularCRC::compute(const uint8_t* bytes, size_t bytelen) const {
    // Compute_CRC32 at http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html
    uint64_t crc = 0;
    for (size_t i = 0; i < bytelen; i++) {
        uint64_t msb = bytes[i];
        msb <<= (d_polylen - 8);
        msb ^= crc;
        size_t tidx = msb >> (d_polylen - 8);
        crc = (crc << 8) ^ d_table[tidx];
        crc &= (1ULL << d_polylen) - 1;
    }
    return crc;
}


}

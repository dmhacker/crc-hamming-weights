#include <crcham/crc.hpp>

namespace crcham {

__device__ 
NaiveCRC::NaiveCRC(uint64_t koopman)
    : d_generator(koopman)
{
    d_length = 64 - __clzll(koopman);
    if (d_length > 0) {
        d_generator ^= 1ULL << (d_length - 1);
    }
    d_generator <<= 1;
    d_generator |= 1;
}

 __device__
TabularCRC::TabularCRC(uint64_t koopman) 
    : d_generator(koopman)
{
    d_length = 64 - __clzll(koopman);
    if (d_length > 0) {
        d_generator ^= 1ULL << (d_length - 1);
    }
    d_generator <<= 1;
    d_generator |= 1;
    // TODO: Make table using http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html
}

}

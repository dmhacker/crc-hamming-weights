#ifndef CRCHAM_FIXED_WIDTH_INTEGER
#define CRCHAM_FIXED_WIDTH_INTEGER

#include <crcham/fixed_width_buffer.hpp>

namespace crcham {

class FixedWidthInteger {
    FixedWidthBuffer& d_buffer;

public:
    __device__
    FixedWidthInteger(FixedWidthBuffer&);
    __device__
        FixedWidthInteger&
        operator=(const FixedWidthInteger&);
    __device__ size_t trailingZeroes();
    __device__ void operator|=(const FixedWidthInteger&);
    __device__ void operator&=(const FixedWidthInteger&);
    __device__ void operator>>=(size_t);
    __device__ void increment();
    __device__ void decrement();
    __device__ void invert();
    __device__ void negate();
};

}

#endif

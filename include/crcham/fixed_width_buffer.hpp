#ifndef CRCHAM_FIXED_WIDTH_BUFFER
#define CRCHAM_FIXED_WIDTH_BUFFER

#include <cstddef>
#include <cstdint>

namespace crcham {

class FixedWidthBuffer {
    size_t d_precision;
    size_t d_size;
    uint64_t d_mask;
    uint64_t* d_arr_p;

public:
    __device__
    FixedWidthBuffer(size_t bit_precision = 128);
    __device__
    FixedWidthBuffer(const FixedWidthBuffer&);
    __device__
        FixedWidthBuffer&
        operator=(FixedWidthBuffer);
    __device__ ~FixedWidthBuffer();

    __device__
    size_t precision() const;
    __device__
    size_t size() const;
    __device__
    uint64_t* get() const;
    __device__
    uint64_t leadingBitMask() const;
};

}

#endif

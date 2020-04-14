#include <crcham/fixed_width_buffer.hpp>

namespace {
template <class T>
__device__ void swap(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}
}

namespace crcham {

__device__
FixedWidthBuffer::FixedWidthBuffer(size_t bit_precision)
    : d_precision(bit_precision)
    , d_size(bit_precision / 64 + 1)
    , d_mask(1)
    , d_arr_p(static_cast<uint64_t*>(malloc(d_size * sizeof(uint64_t))))
{
    d_mask <<= (bit_precision % 64);
    d_mask--;
    memset(d_arr_p, 0, d_size * sizeof(uint64_t));
}

__device__
FixedWidthBuffer::FixedWidthBuffer(const FixedWidthBuffer& buffer)
    : d_precision(buffer.d_precision)
    , d_size(buffer.d_size)
    , d_mask(buffer.d_mask)
    , d_arr_p(static_cast<uint64_t*>(malloc(d_size * sizeof(uint64_t))))
{
    memcpy(d_arr_p, buffer.d_arr_p, d_size * sizeof(uint64_t));
}

__device__
    FixedWidthBuffer&
    FixedWidthBuffer::operator=(FixedWidthBuffer buffer)
{
    swap(d_precision, buffer.d_precision);
    swap(d_size, buffer.d_size);
    swap(d_mask, buffer.d_mask);
    swap(d_arr_p, buffer.d_arr_p);
    return *this;
}

__device__
    FixedWidthBuffer::~FixedWidthBuffer()
{
    free(d_arr_p);
}

__device__
    size_t
    FixedWidthBuffer::precision() const
{
    return d_precision;
}

__device__
    size_t
    FixedWidthBuffer::size() const
{
    return d_size;
}

__device__
    uint64_t*
    FixedWidthBuffer::get() const
{
    return d_arr_p;
}

__device__
    uint64_t 
    FixedWidthBuffer::leadingBitMask() const
{
    return d_mask;
}

__device__
bool FixedWidthBuffer::operator==(const FixedWidthBuffer& buffer) const {
    if (d_precision != buffer.d_precision) {
        return false;
    }
    for (size_t i = 0; i < d_size; i++) {
        if (d_arr_p[i] != buffer.d_arr_p[i]) {
            return false;
        }
    }
    return true;
}

__device__
bool FixedWidthBuffer::operator!=(const FixedWidthBuffer& buffer) const {
    return !(*this == buffer);
}

}

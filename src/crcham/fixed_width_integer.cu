#include <crcham/fixed_width_integer.hpp>

namespace crcham {

__device__
FixedWidthInteger::FixedWidthInteger(FixedWidthBuffer& buffer)
    : d_buffer(buffer)
{
}

__device__
    FixedWidthInteger&
    FixedWidthInteger::operator=(const FixedWidthInteger& fwint)
{
    // Copy fwint's buffer into our own.
    // NOTE: this assumes that our buffer is the same size as fwint's buffer
    memcpy(d_buffer.get(), fwint.d_buffer.get(), d_buffer.size() * sizeof(uint64_t));
    return *this;
}

__device__ size_t FixedWidthInteger::trailingZeroes()
{
    return 0;
}

__device__ void FixedWidthInteger::operator|=(const FixedWidthInteger& fwint)
{
}

__device__ void FixedWidthInteger::operator&=(const FixedWidthInteger& fwint)
{
}

__device__ void FixedWidthInteger::operator>>=(size_t shifts)
{
}

__device__ void FixedWidthInteger::increment()
{
}

__device__ void FixedWidthInteger::decrement()
{
}

__device__ void FixedWidthInteger::invert()
{
}

__device__ void FixedWidthInteger::negate()
{
}

}

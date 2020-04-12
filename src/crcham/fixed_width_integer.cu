#include <crcham/fixed_width_integer.hpp>

namespace crcham {

namespace {
    __device__
    size_t ctz64(uint64_t x) 
    {
        // Assume x is not 0, otherwise this is undefined
        size_t result = 1;
        if ((x & 0xFFFFFFFF) == 0) {
            result += 32; 
            x = x >> 32;
        }
        if ((x & 0x0000FFFF) == 0) {
            result += 16; 
            x = x >>16;
        }
        if ((x & 0x000000FF) == 0) {
            result += 8; 
            x = x >> 8;
        }
        if ((x & 0x0000000F) == 0) {
            result += 4; 
            x = x >> 4;
        }
        if ((x & 0x00000003) == 0) {
            result += 2;
            x = x >> 2;
        }
        return result - (x & 1);
    }
}

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

__device__ 
size_t FixedWidthInteger::trailingZeroes()
{
    auto ptr = d_buffer.get();
    for (size_t i = 1; i < d_buffer.size(); i++) {
        size_t j = d_buffer.size() - 1 - i;
        if (ptr[j] != 0) {
            return (i - 1) * 64 + ctz64(ptr[j]);
        }
    }
    return (d_buffer.size() - 1) * 64 + umin(ctz64(ptr[0]), d_buffer.precision() % 64);
}

__device__ 
void FixedWidthInteger::operator|=(const FixedWidthInteger& other)
{
    auto ptr = d_buffer.get();
    auto optr = other.d_buffer.get();
    for (size_t i = 0; i < d_buffer.size(); i++) {
        ptr[i] |= optr[i];
    }
    ptr[0] &= d_buffer.leadingBitMask();

}

__device__ 
void FixedWidthInteger::operator&=(const FixedWidthInteger& other)
{
    auto ptr = d_buffer.get();
    auto optr = other.d_buffer.get();
    for (size_t i = 0; i < d_buffer.size(); i++) {
        ptr[i] &= optr[i];
    }
    ptr[0] &= d_buffer.leadingBitMask();

}

__device__ 
void FixedWidthInteger::operator>>=(size_t shifts)
{
    
}

__device__ 
void FixedWidthInteger::increment()
{
    auto ptr = d_buffer.get();
    for (size_t i = 1; i < d_buffer.size(); i++) {
        size_t j = d_buffer.size() - 1 - i;
        if (ptr[j] + 1 == 0) {
            ptr[j] = 0;
            return;
        }
        else {
            ptr[j]++;
        }
    }
    if (ptr[0] == d_buffer.leadingBitMask()) {
        ptr[0] = 0;
    }
    else {
        ptr[0]++;
    }
}

__device__ 
void FixedWidthInteger::decrement()
{
    auto ptr = d_buffer.get();
    for (size_t i = 1; i < d_buffer.size(); i++) {
        size_t j = d_buffer.size() - 1 - i;
        if (ptr[j] == 0) {
            ptr[j]--;
        }
        else {
            ptr[j]--;
            return;
        }
    }
    if (ptr[0] == 0) {
        ptr[0] = d_buffer.leadingBitMask();
    }
    else {
        ptr[0]--;
    }
}

__device__ 
void FixedWidthInteger::invert()
{
    auto ptr = d_buffer.get();
    for (size_t i = 0; i < d_buffer.size(); i++) {
        ptr[i] = ~ptr[i];
    }
    ptr[0] &= d_buffer.leadingBitMask();
}

__device__ 
void FixedWidthInteger::negate()
{
    invert();
    decrement();
}

__device__ 
void FixedWidthInteger::nextPermutation(FixedWidthInteger& perm, 
    FixedWidthInteger& tmp1, FixedWidthInteger& tmp2) 
{
    size_t ptz = perm.trailingZeroes() + 1;
    tmp1 = perm;
    tmp1.decrement();
    perm |= tmp1;
    tmp1 = perm;
    tmp1.invert();
    tmp2 = tmp1;
    tmp2.negate();
    tmp1 &= tmp2;
    tmp1.decrement();
    tmp1 >>= ptz;
    perm.increment();
    perm |= tmp1;
}

}

#include <crcham/fixed_width_integer.hpp>
#include <crcham/integer_operations.hpp>

#include <stdio.h>

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

__device__ 
size_t FixedWidthInteger::trailingZeroes()
{
    auto ptr = d_buffer.get();
    for (size_t i = 0; i < d_buffer.size() - 1; i++) {
        size_t j = d_buffer.size() - 1 - i;
        if (ptr[j] != 0) {
            return i * 64 + ctz64(ptr[j]);
        }
    }
    return (d_buffer.size() - 1) * 64 + umin(ctz64(ptr[0]), d_buffer.precision() % 64);
}

__device__ 
size_t FixedWidthInteger::hammingWeight() {
    auto ptr = d_buffer.get();
    size_t ones = 0;
    for (size_t i = 0; i < d_buffer.size(); i++) {
        ones += __popcll(ptr[i]);
    }
    return ones;
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
    shifts = umin(shifts, d_buffer.precision());
    size_t element_shifts = shifts / 64; 
    size_t bit_shifts = shifts % 64;
    auto ptr = d_buffer.get();
    for (size_t i = 0; i < d_buffer.size() - element_shifts; i++) {
        size_t to = d_buffer.size() - i - 1;
        size_t from = to - element_shifts;
        uint64_t previous = from == 0 ? 0 : ptr[from - 1];
        ptr[to] = (ptr[from] >> bit_shifts) | (previous << (64 - bit_shifts));
    }
    for (size_t i = 0; i < element_shifts; i++) {
        ptr[i] = 0;
    }
    ptr[0] &= d_buffer.leadingBitMask();
}

__device__ 
void FixedWidthInteger::increment()
{
    auto ptr = d_buffer.get();
    for (size_t i = 0; i < d_buffer.size() - 1; i++) {
        size_t j = d_buffer.size() - 1 - i;
        if (ptr[j] + 1 == 0) {
            ptr[j] = 0;
        }
        else {
            ptr[j]++;
            return;
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
    for (size_t i = 0; i < d_buffer.size() - 1; i++) {
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
    increment();
}

__device__ 
void FixedWidthInteger::permuteNext(FixedWidthInteger& tmp1, 
    FixedWidthInteger& tmp2) 
{
    auto& perm = *this;
    size_t ptz = trailingZeroes() + 1;
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

__device__ 
void FixedWidthInteger::permuteNth(uint64_t n, size_t k) {
    size_t mmax = d_buffer.precision();
    uint64_t* ptr = d_buffer.get();
    size_t offset;
    for (offset = 0; offset < d_buffer.size(); offset++) {
        ptr[offset] = 0;
    }
    for (size_t i = 0; i < mmax; i++) {
        size_t m = mmax - i;
        uint64_t m1ck = ncr64(m - 1, k); 
        if (i < mmax % 64) {
            offset = 0; 
        }
        else {
            offset = (i - mmax % 64) / 64 + 1;
        }
        ptr[offset] <<= 1;
        if (n >= m1ck) {
            ptr[offset] |= 1;
            n -= m1ck;
            k--;
        }
    }
}

}

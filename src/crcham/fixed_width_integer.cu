#include <crcham/fixed_width_integer.hpp>

#include <stdio.h>

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

    __device__
    size_t choose(size_t n, size_t k) {
        // https://stackoverflow.com/questions/3025162/statistics-combinations-in-python 
        // Assume that k <= n
        size_t ntok = 1;
        size_t ktok = 1;
        size_t tmax = umin(k, n - k);
        for (size_t t = 1; t <= tmax; t++) {
            ntok *= n;
            ktok *= t;
            n -= 1;
        }
        return ntok;
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
void FixedWidthInteger::permuteNth(size_t n, size_t k) {
    // TODO: Fill buffer
    // TODO: Use uint64_t instead of size_t?
    size_t m = d_buffer.precision();
    size_t mx = choose(m - 1, k); 
    for (size_t i = 0; i < m; i++) {
        if (n < mx) {
            // ith bit is a 0
        }
        else {
            // ith bit is a 1
            n -= mx;
            k--;
            mx = choose(m - 1, k);
        }
    }
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


}

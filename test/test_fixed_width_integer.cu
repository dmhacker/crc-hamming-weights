#include "tests.hpp"

#include <crcham/fixed_width_integer.hpp>

using namespace crcham;

__global__
void testFWIIncrement(size_t prec) {
    FixedWidthBuffer buffer(prec);
    FixedWidthInteger fwint(buffer); 
    // Underflow test: start at 0, increment 3 times
    for (int i = 0; i < 3; i++)
        fwint.increment();
    for (size_t i = 0; i < buffer.size() - 1; i++) {
        assert(buffer.get()[i] == 0);
    }
    assert(buffer.get()[buffer.size() - 1] == 3);
    // Overflow test: start at maximum number, increment once
    for (size_t i = 1; i < buffer.size(); i++) {
        buffer.get()[i] = 0;
        buffer.get()[i]--;
    }
    buffer.get()[0] = buffer.leadingBitMask();
    fwint.increment();
    for (size_t i = 0; i < buffer.size(); i++) {
        assert(buffer.get()[i] == 0);
    }
}

__global__
void testFWIDecrement(size_t prec) {
    FixedWidthBuffer buffer(prec);
    FixedWidthInteger fwint(buffer); 
    // Overflow test: start at 0, decrement once
    uint64_t maximum = 0;
    maximum--;
    fwint.decrement();
    for (size_t i = 1; i < buffer.size(); i++) {
        assert(buffer.get()[i] == maximum);
    }
    assert(buffer.get()[0] == buffer.leadingBitMask());
    // Underflow test: check that borrowing works
    if (buffer.precision() >= 65) {
        memset(buffer.get(), 0, sizeof(uint64_t) * buffer.size());
        buffer.get()[buffer.size() - 2] = 1;
        buffer.get()[buffer.size() - 1] = 1;
        fwint.decrement();
        for (size_t i = 0; i < buffer.size() - 2; i++) {
            assert(buffer.get()[i] == 0);
        }
        assert(buffer.get()[buffer.size() - 2] == 1);
        assert(buffer.get()[buffer.size() - 1] == 0);
        fwint.decrement();
        for (size_t i = 0; i < buffer.size() - 1; i++) {
            assert(buffer.get()[i] == 0);
        }
        assert(buffer.get()[buffer.size() - 1] == maximum);
        fwint.decrement();
        for (size_t i = 0; i < buffer.size() - 1; i++) {
            assert(buffer.get()[i] == 0);
        }
        assert(buffer.get()[buffer.size() - 1] == maximum - 1);
    }
}

__global__
void testFWIInvert(size_t prec) {
    FixedWidthBuffer buffer(prec);
    FixedWidthInteger fwint(buffer); 
    // Inverting a number with all 0's should give us all 1's
    uint64_t maximum = 0;
    maximum--;
    fwint.invert();
    for (size_t i = 1; i < buffer.size(); i++) {
        assert(buffer.get()[i] == maximum);
    }
    assert(buffer.get()[0] == buffer.leadingBitMask());
    // Inverting again should give us back all of the 0's
    fwint.invert();
    for (size_t i = 0; i < buffer.size(); i++) {
        assert(buffer.get()[i] == 0);
    }
    // Create alternating bit pattern 0101... and invert
    for (size_t i = 1; i < buffer.size(); i++) {
        buffer.get()[i] = 0x5555555555555555;
    }
    buffer.get()[0] = buffer.leadingBitMask() & 0x5555555555555555;
    fwint.invert();
    for (size_t i = 1; i < buffer.size(); i++) {
        assert(buffer.get()[i] == 0xAAAAAAAAAAAAAAAA);
    }
    assert(buffer.get()[0] == (buffer.leadingBitMask() & 0xAAAAAAAAAAAAAAAA));
}

__global__
void testFWIAnd(size_t prec) {
    // TODO: Implement this function
}

__global__
void testFWIOr(size_t prec) {
    // TODO: Implement this function
}

__global__
void testFWITrailingZeroes(size_t prec, size_t zeroes) {
    FixedWidthBuffer buffer(prec);
    FixedWidthInteger fwint(buffer); 
    if (prec != zeroes) {
        size_t skips = zeroes / 64;
        size_t shifts = zeroes % 64; 
        buffer.get()[buffer.size() - 1 - skips] = 1 << shifts;
    }
    assert(fwint.trailingZeroes() == zeroes);
}

__global__
void testFWIRightShift(size_t prec, size_t shifts) {
    // TODO: Implement this function
}

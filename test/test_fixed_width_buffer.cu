#include "all_tests.hpp"

#include <crcham/fixed_width_buffer.hpp>

using namespace crcham;

__device__
void testFWBMetadata(size_t prec, size_t sz, uint64_t mask) {
    FixedWidthBuffer buffer(prec);
    assert(buffer.precision() == prec);
    assert(buffer.size() == sz);
    assert(buffer.leadingBitMask() == mask);
    for (size_t i = 0; i < buffer.size(); i++) {
        assert(buffer.get()[i] == 0);
    }
}

__device__
void testFWBEquality(size_t prec) {
    FixedWidthBuffer buf1(prec);
    FixedWidthBuffer buf2(prec);
    assert(buf1 == buf2);
    buf1.get()[buf1.size() - 1] = 0x1;
    buf2.get()[buf2.size() - 1] = 0x1;
    assert(buf1 == buf2);
}

__device__
void testFWBInequality(size_t prec) {
    FixedWidthBuffer buf1(prec);
    FixedWidthBuffer buf2(prec);
    buf1.get()[buf1.size() - 1] = 0xF;
    buf2.get()[buf2.size() - 1] = 0xA;
    assert(buf1 != buf2);
}

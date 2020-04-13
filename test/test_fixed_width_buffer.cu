#include "tests.hpp"

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
    // TODO
}

__device__
void testFWBInequality(size_t prec) {
    // TODO
}

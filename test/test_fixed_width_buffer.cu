#include "tests.hpp"

#include <crcham/fixed_width_buffer.hpp>

using namespace crcham;

__global__
void testFWBuffer(size_t prec, size_t sz, uint64_t mask) {
    FixedWidthBuffer buffer(prec);
    assert(buffer.precision() == prec);
    assert(buffer.size() == sz);
    assert(buffer.leadingBitMask() == mask);
    for (size_t i = 0; i < buffer.size(); i++) {
        assert(buffer.get()[i] == 0);
    }
}

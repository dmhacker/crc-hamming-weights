#ifndef TESTS_HPP
#define TESTS_HPP

#include <iostream>
#include <cassert>

__global__
void testFWBuffer(size_t prec, size_t sz, uint64_t mask);

__global__
void testFWIIncrement(size_t prec);
__global__
void testFWIDecrement(size_t prec);
__global__
void testFWIInvert(size_t prec);
__global__
void testFWIAnd(size_t prec);
__global__
void testFWIOr(size_t prec);
__global__
void testFWITrailingZeroes(size_t prec, size_t zeroes);
__global__
void testFWIRightShift(size_t prec, size_t shifts);


#endif

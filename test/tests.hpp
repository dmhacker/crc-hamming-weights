#ifndef TESTS_HPP
#define TESTS_HPP

#include <iostream>
#include <cassert>

__device__
void testFWBMetadata(size_t prec, size_t sz, uint64_t mask);

__device__
void testFWIIncrement(size_t prec);
__device__
void testFWIDecrement(size_t prec);
__device__
void testFWIInvert(size_t prec);
__device__
void testFWIAnd(size_t prec);
__device__
void testFWIOr(size_t prec);
__device__
void testFWITrailingZeroes(size_t prec, size_t zeroes);
__device__
void testFWIRightShift(size_t prec, size_t shifts);
__device__
void testFWIPermuteNext(size_t prec, size_t weight);


#endif

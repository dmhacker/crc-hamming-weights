#ifndef CRCHAM_COMPUTE_HPP
#define CRCHAM_COMPUTE_HPP

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <crcham/crc.hpp>

namespace crcham {

size_t hammingWeightGPU(float* timing, uint64_t polynomial, size_t message_bits, size_t error_bits);
size_t hammingWeightCPU(float* timing, uint64_t polynomial, size_t message_bits, size_t error_bits);

}

#endif

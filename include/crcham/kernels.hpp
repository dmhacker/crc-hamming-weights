#ifndef CRCHAM_KERNELS_HPP
#define CRCHAM_KERNELS_HPP

#include <cassert>
#include <cstdio>

#include <crcham/codeword.hpp>
#include <crcham/crc.hpp>
#include <crcham/math.hpp>

namespace crcham {

template <class CRC>
__global__
void hammingWeight(size_t* weights, CRC crc, size_t message_bits, size_t error_bits);

}

#endif

#include <iostream>

#include <crcham/fixed_width_integer.hpp>

using namespace crcham;

__global__
void testKernel(size_t message_size, size_t weight) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x; 
    FixedWidthBuffer codeword(message_size);
    FixedWidthInteger permutation(codeword);
    {
        uint64_t maximum = 0;
        maximum--;
        size_t maxes = weight / 64;
        for (size_t i = 0; i < maxes; i++) {
            codeword.get()[codeword.size() - 1 - i] = maximum;
        }
        size_t remaining = weight % 64;
        if (remaining != 0) {
            codeword.get()[codeword.size() - 1 - maxes] = (1ULL << remaining) - 1;
        }
    }
    FixedWidthBuffer buf1(codeword.precision());
    FixedWidthBuffer buf2(codeword.precision());
    FixedWidthInteger tmp1(buf1);
    FixedWidthInteger tmp2(buf2);
    for (size_t i = 0; i < index; i++) {
        FixedWidthInteger::permute(permutation, tmp1, tmp2);
    }
    for (size_t iter = 0; iter < 10; iter++) {
        // TODO: Perform CRC checks inside here
        for (size_t i = 0; i < blockDim.x * gridDim.x; i++) {
            FixedWidthInteger::permute(permutation, tmp1, tmp2);
        }
    }
}

int main()
{
    int devices = 0;
    cudaGetDeviceCount(&devices);
    if (devices == 0) {
        std::cerr << "Unable to find a CUDA-compatible GPU." << std::endl;
        return EXIT_FAILURE;
    }

    testKernel<<<2, 1024>>>(6127, 3); 
    cudaSetDeviceFlags(cudaDeviceBlockingSync);
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}

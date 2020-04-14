#include <iostream>
#include <cassert>

#include <crcham/integer_operations.hpp>
#include <crcham/fixed_width_integer.hpp>

using namespace crcham;

__global__
void testKernel(size_t message_size, size_t weight) {
    uint64_t thread_count = gridDim.x * blockDim.x; 
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; 
    uint64_t max_combinations = ncr64(message_size, weight);
    FixedWidthBuffer codeword(message_size);
    FixedWidthInteger permutation(codeword);
    for (; index < max_combinations; index += thread_count) {
        permutation.permuteNth(index, weight);
        assert(permutation.hammingWeight() == weight); 
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
    cudaDeviceProp device0;
    cudaGetDeviceProperties(&device0, 0);
    std::cout << "Found CUDA device: " << device0.name << std::endl;

    testKernel<<<5, 1024>>>(1056, 4); 
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}

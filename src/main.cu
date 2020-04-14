#include <iostream>
#include <cassert>

#include <crcham/operations.hpp>
#include <crcham/codeword.hpp>

using namespace crcham;

__global__
void testKernel(size_t message_size, size_t weight) {
    uint64_t thread_count = gridDim.x * blockDim.x; 
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; 
    uint64_t max_combinations = ncr64(message_size, weight);
    Codeword<32> codeword;
    for (; index < max_combinations; index += thread_count) {
        codeword.permute(index, message_size, weight);
        assert(codeword.popcount() == weight); 
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

    testKernel<<<5, 1024>>>(201, 4); 
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}

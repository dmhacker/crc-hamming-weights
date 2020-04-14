#include <iostream>
#include <cassert>

#include <crcham/operations.hpp>
#include <crcham/codeword.hpp>

using namespace crcham;

__global__
void testKernel(size_t message_size, size_t weight) {
    Codeword<64> codeword;
    size_t tcnt = gridDim.x * blockDim.x; 
    size_t tid = threadIdx.x; 
    uint64_t index = blockIdx.x * blockDim.x + tid; 
    uint64_t max_combinations = ncr64(message_size, weight);
    for (; index < max_combinations; index += tcnt) {
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
    cudaSetDeviceFlags(cudaDeviceBlockingSync);

    testKernel<<<512, 512>>>(500, 4); 
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}

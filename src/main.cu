#include <iostream>
#include <cassert>

#include <crcham/operations.hpp>
#include <crcham/codeword.hpp>

using namespace crcham;

__global__
void testKernel(size_t message_size, size_t weight) {
    extern __shared__ Codeword<64> codewords[];
    uint64_t thread_count = gridDim.x * blockDim.x; 
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; 
    uint64_t max_combinations = ncr64(message_size, weight);
    for (; index < max_combinations; index += thread_count) {
        codewords[threadIdx.x].permute(index, message_size, weight);
        assert(codewords[threadIdx.x].popcount() == weight); 
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

    testKernel<<<512, 512, 512 * sizeof(Codeword<64>)>>>(300, 4); 
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}

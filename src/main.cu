#include <iostream>
#include <cassert>

#include <crcham/codeword.hpp>
#include <crcham/cuda.hpp>
#include <crcham/operations.hpp>

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
    CUDA cuda;
    if (!cuda.enabled()) {
        std::cerr << "A supported NVIDIA GPU could not be found." << std::endl;
        return EXIT_FAILURE;
    }
    cuda.setup();
    std::cout << cuda;

    testKernel<<<512, 512>>>(500, 4); 
    cuda.wait();

    return EXIT_SUCCESS;
}

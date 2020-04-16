#include <iostream>
#include <cassert>

#include <crcham/codeword.hpp>
#include <crcham/cuda.hpp>
#include <crcham/operations.hpp>

using namespace crcham;

__global__
void testKernel(size_t message_size, size_t weight) {
    Codeword<64> codeword;
    size_t tid = threadIdx.x; 
    size_t pincr = gridDim.x * blockDim.x; 
    uint64_t pidx = blockIdx.x * blockDim.x + tid; 
    uint64_t pmax = ncr64(message_size, weight);
    for (; pidx < pmax; pidx += pincr) {
        codeword.permute(pidx, message_size, weight);
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

    testKernel<<<512, 512>>>(250, 4); 
    cuda.wait();

    return EXIT_SUCCESS;
}

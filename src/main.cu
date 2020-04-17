#include <iostream>
#include <cassert>

#include <crcham/codeword.hpp>
#include <crcham/permute.hpp>
#include <crcham/math.hpp>

using namespace crcham;

__global__
void testKernel(size_t message_size, size_t weight) {
    /* size_t buflen = message_size / 32; */
    /* if (message_size % 32 != 0) { */
    /*     buflen++; */
    /* } */
    /* auto buffer = static_cast<uint32_t*>(malloc(buflen * sizeof(uint32_t))); */
    uint32_t buffer[10];

    size_t tid = threadIdx.x; 
    size_t pincr = gridDim.x * blockDim.x; 
    uint64_t pidx = blockIdx.x * blockDim.x + tid; 
    uint64_t pmax = ncrll(message_size, weight);

    for (; pidx < pmax; pidx += pincr) {
        permute(buffer, 10, pidx, message_size, weight);
        assert(popcount(buffer, 10) == weight); 
    }

    /* free(buffer); */
}

int main()
{
    size_t message_size = 300;
    size_t hamming_weight = 4;

    // Check that there is an available CUDA device
    {
        int devcnt = 0;
        cudaGetDeviceCount(&devcnt);
        if (devcnt == 0) {
            std::cerr << "A supported NVIDIA GPU could not be found." << std::endl;
            return EXIT_FAILURE;
        }
    }

    // CPU should not busy-wait for the kernel to finish
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    // Find optimal block and grid sizes
    int grid_size;
    int block_size;
    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, testKernel);

    // Set maximum allowable memory sizes
    size_t original_heap;
    size_t required_heap = 2 * grid_size * block_size * (message_size / 8);
    cudaDeviceGetLimit(&original_heap, cudaLimitMallocHeapSize);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, std::max(original_heap, required_heap));

    // Run the kernel and block until it is done
    testKernel<<<grid_size, block_size>>>(message_size, hamming_weight); 
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}

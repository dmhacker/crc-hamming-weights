#include <iostream>
#include <cassert>

#include <crcham/codeword.hpp>
#include <crcham/permute.hpp>
#include <crcham/math.hpp>
#include <crcham/kernels.hpp>

int main()
{
    uint64_t polynomial = 0xa; 
    size_t message_bits = 300;
    size_t error_bits = 4;

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
    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, 
        crcham::hammingWeight<crcham::TabularCRC>);

    // Set maximum allowable memory sizes
    size_t original_heap;
    size_t required_heap = 2 * grid_size * block_size * (message_bits / 8);
    cudaDeviceGetLimit(&original_heap, cudaLimitMallocHeapSize);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, std::max(original_heap, required_heap));

    // Allocate memory for thread-local weights
    size_t* weights;
    cudaMallocManaged(&weights, grid_size * block_size * sizeof(size_t));
    cudaMemset(weights, 0, grid_size * block_size * sizeof(size_t));

    // Run the kernel and block until it is done
    crcham::NaiveCRC ncrc(polynomial);
    if (ncrc.length() < 8) {
        crcham::hammingWeight<crcham::NaiveCRC><<<grid_size, block_size>>>(weights, ncrc, message_bits, error_bits); 
    }
    else {
        crcham::TabularCRC tcrc(polynomial);
        crcham::hammingWeight<crcham::TabularCRC><<<grid_size, block_size>>>(weights, tcrc, message_bits, error_bits); 
    }
    cudaDeviceSynchronize();

    // Accumulate results from all threads
    size_t weight = 0;
    for (size_t i = 0; i < grid_size * block_size; i++) {
        weight += weights[i];
    }
    std::cout << "Hamming Weight = " << weight << std::endl;

    return EXIT_SUCCESS;
}

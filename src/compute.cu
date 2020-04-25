#include <iostream>

#include <crcham/compute.hpp>
#include <crcham/codeword.hpp>
#include <crcham/crc.hpp>
#include <crcham/math.hpp>

namespace crcham {

namespace {

template <class CRC>
__global__
void hammingWeightKernel(size_t* weights, CRC crc, size_t message_bits, size_t error_bits) {
    // Allocate the minimum number of integers required to hold the message and FCS field
    size_t codeword_bits = message_bits + crc.length();
    size_t codeword_bytes = codeword_bits / 8;
    if (codeword_bits % 8 != 0) {
        codeword_bytes++;
    }
    auto codeword_byte_ptr = static_cast<uint8_t*>(
        malloc(codeword_bytes * sizeof(uint8_t)));

    const size_t widx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pincr = gridDim.x * blockDim.x; 
    uint64_t pidx = blockIdx.x * blockDim.x + threadIdx.x; 
    uint64_t pmax = ncrll(codeword_bits, error_bits);
    size_t weight = 0;

    for (; pidx < pmax; pidx += pincr) {
        // Permute the bytes in the ${pidx}th way
        permute(codeword_byte_ptr, codeword_bytes, pidx, codeword_bits, error_bits);
        assert(popcount(codeword_byte_ptr, codeword_bytes) == error_bits); 
        // Test to see if the codeword with errors is considered valid
        uint64_t error_crc = extract(codeword_byte_ptr, codeword_bytes, codeword_bits, crc.length());
        uint64_t good_crc = crc.compute(codeword_byte_ptr, codeword_bytes);
        if (error_crc == good_crc) {
            weight++;
        }
    }
    weights[widx] = weight;

    free(codeword_byte_ptr);
}

}

size_t hammingWeightGPU(float* timing, uint64_t polynomial, size_t message_bits, size_t error_bits) 
{
    // Check that there is an available CUDA device
    {
        int devcnt = 0;
        cudaGetDeviceCount(&devcnt);
        if (devcnt == 0) {
            throw std::runtime_error("A supported NVIDIA GPU could not be found.");
        }
    }

    // CPU should not busy-wait for the kernel to finish
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    // Find optimal block and grid sizes
    int grid_size;
    int block_size;
    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, 
        crcham::hammingWeightKernel<crcham::TabularCRC>);

    // Set maximum allowable memory sizes
    size_t original_heap;
    size_t required_heap = 2 * grid_size * block_size * (message_bits / 8);
    cudaDeviceGetLimit(&original_heap, cudaLimitMallocHeapSize);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 
            std::max(original_heap, required_heap));

    // Allocate memory for thread-local weights
    size_t* weights;
    cudaMallocManaged(&weights, grid_size * block_size * sizeof(size_t));
    cudaMemset(weights, 0, grid_size * block_size * sizeof(size_t));

    // Run the kernel and block until it is done
    cudaEvent_t start_event; 
    cudaEvent_t stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event);
    size_t polylen = crcham::NaiveCRC(polynomial).length();
    if (polylen < 8) {
        crcham::NaiveCRC ncrc(polynomial);
        crcham::hammingWeightKernel<crcham::NaiveCRC><<<grid_size, block_size>>>(
                weights, ncrc, message_bits, error_bits); 
    }
    else {
        crcham::TabularCRC tcrc(polynomial);
        crcham::hammingWeightKernel<crcham::TabularCRC><<<grid_size, block_size>>>(
                weights, tcrc, message_bits, error_bits); 
    }
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(timing, start_event, stop_event);

    // Accumulate results from all threads
    size_t weight = 0;
    for (size_t i = 0; i < grid_size * block_size; i++) {
        weight += weights[i];
    }
    cudaFree(weights);

    return weight;
}

size_t hammingWeightCPU(float* timing, uint64_t polynomial, size_t message_bits, size_t error_bits) 
{
    throw std::runtime_error("Unimplemented.");
}

}

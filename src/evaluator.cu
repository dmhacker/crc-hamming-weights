#include <stdexcept>
#include <thread>

#include <crcham/codeword.hpp>
#include <crcham/crc.hpp>
#include <crcham/evaluator.hpp>
#include <crcham/math.hpp>

#include <omp.h>

namespace crcham {

namespace {

template <class CRC>
__global__
void weightsKernel(size_t* weights, CRC crc, size_t message_bits, size_t error_bits) {
    size_t codeword_bits = message_bits + crc.length();
    size_t codeword_bytes = codeword_bits / 8;
    if (codeword_bits % 8 != 0) {
        codeword_bytes++;
    }
    auto codeword = static_cast<uint8_t*>(
        malloc(codeword_bytes * sizeof(uint8_t)));

    const size_t widx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pincr = gridDim.x * blockDim.x; 
    uint64_t pidx = blockIdx.x * blockDim.x + threadIdx.x; 
    uint64_t pmax = ncrll(codeword_bits, error_bits);
    size_t weight = 0;

    for (; pidx < pmax; pidx += pincr) {
        permute(codeword, codeword_bytes, pidx, codeword_bits, error_bits);
        uint64_t error_crc = extract(codeword, codeword_bytes, codeword_bits, crc.length());
        uint64_t good_crc = crc.compute(codeword, codeword_bytes);
        if (error_crc == good_crc) {
            weight++;
        }
    }
    weights[widx] = weight;

    free(codeword);
}

template <class CRC>
size_t weightsOpenMP(const CRC& crc, size_t message_bits, size_t error_bits) 
{
    size_t codeword_bits = message_bits + crc.length();
    size_t codeword_bytes = codeword_bits / 8;
    if (codeword_bits % 8 != 0) {
        codeword_bytes++;
    }

    auto num_threads = std::max(3u, std::thread::hardware_concurrency()) - 2;
    auto codewords = new uint8_t[num_threads * codeword_bytes]();
    auto weights = new size_t[num_threads]();
    uint64_t pmax = ncrll(codeword_bits, error_bits);

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t pidx = 0; pidx < pmax; pidx++) {
        auto codeword = codewords + codeword_bytes * omp_get_thread_num();
        permute(codeword, codeword_bytes, pidx, codeword_bits, error_bits);
        uint64_t error_crc = extract(codeword, codeword_bytes, codeword_bits, crc.length());
        uint64_t good_crc = crc.compute(codeword, codeword_bytes);
        if (error_crc == good_crc) {
            weights[omp_get_thread_num()]++;
        }
    }

    delete[] codewords;
    size_t weight = 0;
    for (size_t i = 0; i < num_threads; i++) {
        weight += weights[i];
    }
    return weight;
}

}

WeightsEvaluator::WeightsEvaluator(uint64_t polynomial, size_t message_bits, size_t error_bits) 
    : d_polynomial(polynomial)
    , d_polylen(crcham::NaiveCRC(polynomial).length())
    , d_message(message_bits)
    , d_errors(error_bits)
    , d_evaluations(crcham::ncrll(message_bits + d_polylen, error_bits))
{
}

template<>
void WeightsEvaluator::run<true>()
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
        crcham::weightsKernel<crcham::TabularCRC>);

    // Set maximum allowable memory sizes
    size_t original_heap;
    size_t required_heap = 2 * grid_size * block_size * (d_message / 8);
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
    if (d_polylen < 8) {
        crcham::NaiveCRC ncrc(d_polynomial);
        crcham::weightsKernel<crcham::NaiveCRC><<<grid_size, block_size>>>(
                weights, ncrc, d_message, d_errors); 
    }
    else {
        crcham::TabularCRC tcrc(d_polynomial);
        crcham::weightsKernel<crcham::TabularCRC><<<grid_size, block_size>>>(
                weights, tcrc, d_message, d_errors); 
    }
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    float millis = 0;
    cudaEventElapsedTime(&millis, start_event, stop_event);
    d_elapsed = std::chrono::milliseconds((unsigned long) millis);

    // Accumulate results from all threads
    d_weight = 0;
    for (size_t i = 0; i < grid_size * block_size; i++) {
        d_weight += weights[i];
    }
    cudaFree(weights);
}

template<>
void WeightsEvaluator::run<false>()
{
    auto timestamp = std::chrono::steady_clock::now();
    if (d_polylen < 8) {
        crcham::NaiveCRC ncrc(d_polynomial);
        d_weight = weightsOpenMP(ncrc, d_message, d_errors);
    }
    else {
        crcham::TabularCRC tcrc(d_polynomial);
        d_weight = weightsOpenMP(tcrc, d_message, d_errors);
    }
    d_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - timestamp);
}

size_t WeightsEvaluator::evaluations() const {
    return d_evaluations;
}

size_t WeightsEvaluator::weight() const {
    return d_weight;
}

std::chrono::milliseconds WeightsEvaluator::elapsed() const {
    return d_elapsed;
}

}

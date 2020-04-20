#include <iostream>
#include <cassert>

#include <crcham/kernels.hpp>

#include <cxxopts.hpp>

int main(int argc, char** argv)
{
    cxxopts::Options options("crcham", "Calculate the hamming weights of CRC polynomials");
    options.add_options()("p,poly", "CRC in Koopman notation (implicit +1)", cxxopts::value<uint64_t>());
    options.add_options()("m,message", "Message length in bits", cxxopts::value<size_t>());
    options.add_options()("e,errors", "Number of bit errors in the message", cxxopts::value<size_t>());
    options.add_options()("h,help", "Print help message");
    auto result = options.parse(argc, argv);
    
    if (result.count("help") > 0) {
        std::cout << options.help();
        return EXIT_SUCCESS;
    }

    uint64_t polynomial;
    try {
        polynomial = result["poly"].as<uint64_t>();
    } catch (std::exception& ex) {
        std::cerr << "Unable to interpret polynomial: " << ex.what() << std::endl;
        std::cerr << "Consider using https://users.ece.cmu.edu/~koopman/crc/crc32.html to select parameters." << std::endl;
        return EXIT_SUCCESS;
    }

    size_t message_bits;
    try {
        message_bits = result["message"].as<size_t>();
    } catch (std::exception& ex) {
        std::cerr << "Unable to interpret message length: " << ex.what() << std::endl;
        std::cerr << "Consider using https://users.ece.cmu.edu/~koopman/crc/crc32.html to select parameters." << std::endl;
        return EXIT_SUCCESS;
    }

    size_t error_bits;
    try {
        error_bits = result["errors"].as<size_t>();
    } catch (std::exception& ex) {
        std::cerr << "Unable to interpret error count: " << ex.what() << std::endl;
        std::cerr << "Consider using https://users.ece.cmu.edu/~koopman/crc/crc32.html to select parameters." << std::endl;
        return EXIT_SUCCESS;
    }

    /********************************** CUDA KERNEL **********************************/ 

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

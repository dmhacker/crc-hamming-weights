#include "test_codeword.hpp"

#include <iostream>

#include <crcham/cuda.hpp>

__global__
void testKernel() {
    testCodewordEqual<1>();
    testCodewordEqual<57>();
    testCodewordEqual<271>();
    testCodewordInequal<1>();
    testCodewordInequal<57>();
    testCodewordInequal<271>();
    for (size_t m = 64; m <= 256; m += 3) {
        for (size_t k = 1; k < 8; k++) {
            // Parameters chosen such that that ((k - 1) choose w) doesn't 
            // exceed the memory limits of a 64-bit unsigned integer
            testCodewordPermute<8>(m, k);
        }
    }
}

int main() {
    crcham::CUDA cuda;
    if (!cuda.enabled()) {
        std::cerr << "A supported NVIDIA GPU could not be found." << std::endl;
        return EXIT_FAILURE;
    }
    cuda.setup();
    std::cout << cuda;

    testKernel<<<1, 1>>>(); 
    std::cout << "Tests started." << std::endl;
    cuda.wait();
    std::cout << "Tests finished." << std::endl;

    return EXIT_SUCCESS;
}

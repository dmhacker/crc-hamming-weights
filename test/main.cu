#include "test_codeword.hpp"

#include <iostream>

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
    int devices = 0;
    cudaGetDeviceCount(&devices);
    if (devices == 0) {
        std::cerr << "Unable to find a CUDA-compatible GPU." << std::endl;
        return EXIT_FAILURE;
    }
    cudaSetDeviceFlags(cudaDeviceBlockingSync);

    testKernel<<<1, 1>>>(); 
    std::cout << "Tests started." << std::endl;

    cudaDeviceSynchronize();
    std::cout << "Tests finished." << std::endl;

    return EXIT_SUCCESS;
}

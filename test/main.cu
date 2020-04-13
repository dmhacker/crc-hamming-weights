#include "tests.hpp"

#include <iostream>

__global__
void testKernel() {
    testFWBMetadata(17, 1, 131071);
    testFWBMetadata(64, 2, 0);
    testFWBMetadata(192, 4, 0);
    testFWBMetadata(201, 4, 511);
    testFWBMetadata(64000, 1001, 0);

    for (size_t p = 64; p <= 256; p += 3) {
        testFWIIncrement(p);
        testFWIDecrement(p);
        testFWIInvert(p);
        testFWIAnd(p);
        testFWIOr(p);
        for (size_t z = 0; z <= p; z++) {
            testFWITrailingZeroes(p, z);
        }
        for (size_t s = 0; s <= p * 2; s++) {
            testFWIRightShift(p, s);
        }
        for (size_t w = 1; w < p; w++) {
            testFWIPermute(p, w);
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

    testKernel<<<1, 1>>>(); 

    cudaDeviceSynchronize();
    std::cout << "Tests finished." << std::endl;
    return EXIT_SUCCESS;
}

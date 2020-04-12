#include "tests.hpp"

#include <iostream>

int main() {
    int devices = 0;
    cudaGetDeviceCount(&devices);
    if (devices == 0) {
        std::cerr << "Unable to find a CUDA-compatible GPU." << std::endl;
        return EXIT_FAILURE;
    }

    testFWBuffer<<<1, 1>>>(17, 1, 131071);
    testFWBuffer<<<1, 1>>>(64, 2, 0);
    testFWBuffer<<<1, 1>>>(192, 4, 0);
    testFWBuffer<<<1, 1>>>(201, 4, 511);
    testFWBuffer<<<1, 1>>>(64000, 1001, 0);

    for (size_t p = 2; p <= 256; p++) {
        testFWIIncrement<<<1, 1>>>(p);
        testFWIDecrement<<<1, 1>>>(p);
        testFWIInvert<<<1, 1>>>(p);
        testFWIAnd<<<1, 1>>>(p);
        testFWIOr<<<1, 1>>>(p);
        for (size_t z = 0; z <= p; z++) {
            testFWITrailingZeroes<<<1, 1>>>(p, z);
        }
        for (size_t s = 0; s <= p * 2; s++) {
            testFWIRightShift<<<1, 1>>>(p, s);
        }
    }

    cudaDeviceSynchronize();
    std::cout << "Tests finished." << std::endl;
    return EXIT_SUCCESS;
}

#include "test_codeword.hpp"
#include "test_crc.hpp"

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
    testCRCMetadata(0xe7, 0xcf, 8);
    testCRCMetadata(0x1abf, 0x157f, 13);
    testCRCMetadata(0x8d95, 0x1b2b, 16);
    testCRCMetadata(0x6fb57, 0x5f6af, 19);
    testCRCMetadata(0x540df0, 0x281be1, 23);
    testCRCMetadata(0x80000d, 0x1b, 24);
    testCRCMetadata(0xad0424f3, 0x5a0849e7, 32);
    testCRCMetadata(0x10000000000d, 0x1b, 45);
    testCRCMetadata(0xd6c9e91aca649ad4, 0xad93d23594c935a9, 64);

    // 3, 4, 5, 6, 7-bit CRC tests
    testCRCCompute<crcham::NaiveCRC>("3T", 0x5, 0x5);
    testCRCCompute<crcham::NaiveCRC>("4T", 0x9, 0x1);
    testCRCCompute<crcham::NaiveCRC>("5T", 0x12, 0x1a);
    testCRCCompute<crcham::NaiveCRC>("6T", 0x33, 0x3c);
    testCRCCompute<crcham::NaiveCRC>("7T", 0x65, 0x50);

    // 8, 16, 32, 64-bit CRC tests
    testCRCCompute<crcham::NaiveCRC>("Test message", 0xe7, 0x6e);
    testCRCCompute<crcham::NaiveCRC>("This is a test", 0xc5db, 0x5fc2);
    testCRCCompute<crcham::NaiveCRC>("Another test", 0xad0424f3, 0x545885e5);
    testCRCCompute<crcham::NaiveCRC>("A fourth test", 0xd6c9e91aca649ad4, 0x802de9d103f28376);
    testCRCCompute<crcham::TabularCRC>("Test message", 0xe7, 0x6e);
    testCRCCompute<crcham::TabularCRC>("This is a test", 0xc5db, 0x5fc2);
    testCRCCompute<crcham::TabularCRC>("Another test", 0xad0424f3, 0x545885e5);
    testCRCCompute<crcham::TabularCRC>("A fourth test", 0xd6c9e91aca649ad4, 0x802de9d103f28376);

    // 11, 14, 27, 30, 35, 56-bit tabular CRC tests

    testCRCCompute<crcham::NaiveCRC>("Test test test", 0x5db, 0x23c);
    testCRCCompute<crcham::NaiveCRC>("all lowercase and extra", 0x2402, 0xf6a);
    testCRCCompute<crcham::NaiveCRC>("2912889378278", 0x5e04635, 0x756a6e);
    testCRCCompute<crcham::NaiveCRC>("AHAHAHAHAHHAHAHAHAHA2891...", 0x31342a2f, 0xcd0d90c);
    testCRCCompute<crcham::NaiveCRC>("Wowweeeee CRC!", 0x400000002, 0xc87d1522);
    testCRCCompute<crcham::NaiveCRC>("Deadbeef?yepp.", 0x8000000000004a, 0xbd82b3c6ff47ca);

    testCRCCompute<crcham::TabularCRC>("Test test test", 0x5db, 0x23c);
    testCRCCompute<crcham::TabularCRC>("all lowercase and extra", 0x2402, 0xf6a);
    testCRCCompute<crcham::TabularCRC>("2912889378278", 0x5e04635, 0x756a6e);
    testCRCCompute<crcham::TabularCRC>("AHAHAHAHAHHAHAHAHAHA2891...", 0x31342a2f, 0xcd0d90c);
    testCRCCompute<crcham::TabularCRC>("Wowweeeee CRC!", 0x400000002, 0xc87d1522);
    testCRCCompute<crcham::TabularCRC>("Deadbeef?yepp.", 0x8000000000004a, 0xbd82b3c6ff47ca);

}

int main() {
    crcham::CUDA cuda;
    if (!cuda.enabled()) {
        std::cerr << "A supported NVIDIA GPU could not be found." << std::endl;
        return EXIT_FAILURE;
    }
    cuda.setup();

    testKernel<<<1, 1>>>(); 
    std::cout << "Tests started." << std::endl;
    cuda.wait();
    std::cout << "Tests finished." << std::endl;

    return EXIT_SUCCESS;
}

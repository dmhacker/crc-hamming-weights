#include "catch.hpp"

#include <cstdint>
#include <cstring>

#include <crcham/crc.hpp>

namespace {

void testCRCMetadata(uint64_t koopman, uint64_t normal, size_t bits)
{
    crcham::NaiveCRC ncrc(koopman);
    REQUIRE(normal == ncrc.normal());
    REQUIRE(bits == ncrc.length());
    crcham::TabularCRC tcrc(koopman);
    REQUIRE(normal == tcrc.normal());
    REQUIRE(bits == tcrc.length());
}

template <class CRC>
void testCRCCompute(const char message[], uint64_t koopman, uint64_t crc)
{
    CRC acrc(koopman);
    uint64_t computed = 
        acrc.compute(reinterpret_cast<const uint8_t*>(message), strlen(message));
    REQUIRE(crc == computed);
}

}

TEST_CASE("CRC translation from Koopman to normal form") {
    testCRCMetadata(0xe7, 0xcf, 8);
    testCRCMetadata(0x1abf, 0x157f, 13);
    testCRCMetadata(0x8d95, 0x1b2b, 16);
    testCRCMetadata(0x6fb57, 0x5f6af, 19);
    testCRCMetadata(0x540df0, 0x281be1, 23);
    testCRCMetadata(0x80000d, 0x1b, 24);
    testCRCMetadata(0xad0424f3, 0x5a0849e7, 32);
    testCRCMetadata(0x10000000000d, 0x1b, 45);
    testCRCMetadata(0xd6c9e91aca649ad4, 0xad93d23594c935a9, 64);
}

TEST_CASE("Shift register CRC computation") {
    SECTION("3, 4, 5, 6, 7-bit CRCs") {
        testCRCCompute<crcham::NaiveCRC>("3T", 0x5, 0x5);
        testCRCCompute<crcham::NaiveCRC>("4T", 0x9, 0x1);
        testCRCCompute<crcham::NaiveCRC>("5T", 0x12, 0x1a);
        testCRCCompute<crcham::NaiveCRC>("6T", 0x33, 0x3c);
        testCRCCompute<crcham::NaiveCRC>("7T", 0x65, 0x50);
    }

    SECTION("8, 16, 32, 64-bit CRCs") {
        testCRCCompute<crcham::NaiveCRC>("Test message", 0xe7, 0x6e);
        testCRCCompute<crcham::NaiveCRC>("This is a test", 0xc5db, 0x5fc2);
        testCRCCompute<crcham::NaiveCRC>("Another test", 0xad0424f3, 0x545885e5);
        testCRCCompute<crcham::NaiveCRC>("A fourth test", 0xd6c9e91aca649ad4, 0x802de9d103f28376);
    }

    SECTION("11, 14, 27, 30, 35, 56-bit CRCs") {
        testCRCCompute<crcham::NaiveCRC>("Test test test", 0x5db, 0x23c);
        testCRCCompute<crcham::NaiveCRC>("all lowercase and extra", 0x2402, 0xf6a);
        testCRCCompute<crcham::NaiveCRC>("2912889378278", 0x5e04635, 0x756a6e);
        testCRCCompute<crcham::NaiveCRC>("AHAHAHAHAHHAHAHAHAHA2891...", 0x31342a2f, 0xcd0d90c);
        testCRCCompute<crcham::NaiveCRC>("Wowweeeee CRC!", 0x400000002, 0xc87d1522);
        testCRCCompute<crcham::NaiveCRC>("Deadbeef?yepp.", 0x8000000000004a, 0xbd82b3c6ff47ca);
    }
}

TEST_CASE("Lookup table CRC computation") {
    SECTION("8, 16, 32, 64-bit CRCs") {
        testCRCCompute<crcham::TabularCRC>("Test message", 0xe7, 0x6e);
        testCRCCompute<crcham::TabularCRC>("This is a test", 0xc5db, 0x5fc2);
        testCRCCompute<crcham::TabularCRC>("Another test", 0xad0424f3, 0x545885e5);
        testCRCCompute<crcham::TabularCRC>("A fourth test", 0xd6c9e91aca649ad4, 0x802de9d103f28376);
    }

    SECTION("11, 14, 27, 30, 35, 56-bit CRCs") {
        testCRCCompute<crcham::TabularCRC>("Test test test", 0x5db, 0x23c);
        testCRCCompute<crcham::TabularCRC>("all lowercase and extra", 0x2402, 0xf6a);
        testCRCCompute<crcham::TabularCRC>("2912889378278", 0x5e04635, 0x756a6e);
        testCRCCompute<crcham::TabularCRC>("AHAHAHAHAHHAHAHAHAHA2891...", 0x31342a2f, 0xcd0d90c);
        testCRCCompute<crcham::TabularCRC>("Wowweeeee CRC!", 0x400000002, 0xc87d1522);
        testCRCCompute<crcham::TabularCRC>("Deadbeef?yepp.", 0x8000000000004a, 0xbd82b3c6ff47ca);
    }
}

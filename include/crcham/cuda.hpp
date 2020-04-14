#ifndef CRCHAM_CUDA_HPP
#define CRCHAM_CUDA_HPP

#include <iostream>
#include <vector>

namespace crcham {

class CUDA {
private:
    std::vector<cudaDeviceProp> d_devices;

public:
    CUDA();

    bool enabled() const;
    void setup() const;
    void wait() const;

private:
    friend std::ostream & operator<<(std::ostream &os, const CUDA& cuda);
};

}

#endif

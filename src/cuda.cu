#include <crcham/cuda.hpp>

namespace crcham {

CUDA::CUDA() {
    int dcnt = 0;
    cudaGetDeviceCount(&dcnt);
    cudaDeviceProp device;
    for (int i = 0; i < dcnt; i++) {
        cudaGetDeviceProperties(&device, i);
        d_devices.push_back(device);
    }
}

bool CUDA::enabled() const {
    return !d_devices.empty();
}

void CUDA::setup() const {
    cudaSetDeviceFlags(cudaDeviceBlockingSync);
}

void CUDA::wait() const {
    cudaDeviceSynchronize();
}

std::ostream & operator<<(std::ostream &os, const CUDA& cuda)
{
    return os << "Found CUDA device: " << cuda.d_devices[0].name << std::endl;
}

}

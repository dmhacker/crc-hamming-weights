#include <crcham/cuda.hpp>

#include <cstdint>

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
    os << "CUDA Device Information:" << std::endl;
    for (size_t i = 0; i < cuda.d_devices.size(); i++) {
        auto& device = cuda.d_devices[i];
        os << "  " << i << ") Name = " << device.name << std::endl
            << "     Compute Capability = " << device.major << "." << device.minor << std::endl
            << "     Streaming Multiprocessors = " << device.multiProcessorCount << std::endl
            << "     Clock Rate = " << device.clockRate << std::endl
            << "     Global Memory = " << device.totalGlobalMem << std::endl;
    }
    return os;
}

}

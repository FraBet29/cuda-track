#include "../include/gpu_params.h"
#include <iostream>
#include <cmath>

void set_gpu_params() {
    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Printing device properties." << std::endl;
    std::cout << "  Device name: " << prop.name << std::endl;
    std::cout << "  Global memory (GB): " << (float) prop.totalGlobalMem / 1.0e9 << std::endl;
    std::cout << "  Shared memory per block (KB) " << (float) prop.sharedMemPerBlock / 1.0e3 << std::endl;
    std::cout << "  Warp-size: " << prop.warpSize << std::endl;
    std::cout << "  Maximum number of threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Memory Clock Rate (MHz): " << prop.memoryClockRate / 1.0e3 << std::endl;
    std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1.0e6 << std::endl;
    // Set global variables
    SHARED_MEMORY_PER_BLOCK = prop.sharedMemPerBlock;
    MAX_THREADS_PER_BLOCK_1D = prop.maxThreadsPerBlock;
    MAX_THREADS_PER_BLOCK_2D = std::floor(std::sqrt(MAX_THREADS_PER_BLOCK_1D));
}
#include "../include/cuda_check.h"

/*
void check_call(cudaError_t call) {
    const cudaError_t err = call;
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);                                                           \
    }       
}
*/

/*
void check_kernel_call() {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);                                                           \
    }          
}
*/
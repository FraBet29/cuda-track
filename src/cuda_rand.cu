#include "../include/cuda_rand.h"

__global__ void setup_kernel(curandState *state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // curand_init (unsigned long long seed, unsigned long long sequence, unsigned long long offset, curandState_t *state)
    curand_init(1234, i, 0, &state[i]);
}
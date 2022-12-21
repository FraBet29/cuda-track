#include "../include/cuda_rand.h"

__global__ void rand_setup_kernel(curandState *state, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = i * blockDim.x; j < N && j < i * (blockDim.x + 1); ++j)
        // curand_init (unsigned long long seed, unsigned long long sequence, unsigned long long offset, curandState_t *state)
        curand_init(1234, j, 0, &state[j]);
}
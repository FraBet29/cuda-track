#include "../include/cuda_rand.h"

__global__ void rand_setup_kernel(curandState *state, int seed, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        // curand_init (unsigned long long seed, unsigned long long sequence, unsigned long long offset, curandState_t *state)
        curand_init(seed, i, 0, &state[i]);
}
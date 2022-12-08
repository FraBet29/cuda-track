#include "../include/cuda_rand.h"

extern __global__ void setup_kernel(curandState *state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, i, 0, &state[i]);
}
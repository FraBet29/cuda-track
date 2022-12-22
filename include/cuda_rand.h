#ifndef CUDA_RAND_H

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define MY_RAND_MAX 0x7fffffff

__global__ void rand_setup_kernel(curandState *state, int N, int seed = 1234);

#define CUDA_RAND_H
#endif
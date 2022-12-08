#ifndef CUDA_RAND_H

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define MY_CUDA_RAND_MAX 0x7fffffff

extern __global__ void setup_kernel(curandState *state);


#define CUDA_RAND_H
#endif
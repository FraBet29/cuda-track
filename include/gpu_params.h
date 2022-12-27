#ifndef GPU_PARAMS_H

extern int SHARED_MEMORY_PER_BLOCK;
extern int MAX_THREADS_PER_BLOCK_1D;
extern int MAX_THREADS_PER_BLOCK_2D;

void set_gpu_params();

#define GPU_PARAMS_H
#endif
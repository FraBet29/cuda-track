#ifndef OPTIM_H
#include <iostream>
#include <vector>
#include <utility>
#include "cuda_variable.h"

struct AdamParams {
    float lr, beta1, beta2, eps, weight_decay;
    static AdamParams get_default();
};

struct CudaAdamVariable {
    float *data, *grad, *m, *v;
    bool decay;
    int size;
    CudaAdamVariable() {};
    CudaAdamVariable(CudaVariable *var, bool decay);
    ~CudaAdamVariable();
};

class Adam {
    AdamParams params;
    int step_count;
    std::vector<CudaAdamVariable> cuda_vars;
public:
    Adam() {};
    Adam(std::vector<std::pair<CudaVariable*, bool>> cuda_vars, AdamParams params);
    ~Adam() {};
    void step();
};

#define OPTIM_H
#endif
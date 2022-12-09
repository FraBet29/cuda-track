#ifndef GCN_H
#include <vector>
#include <utility>
#include "variable.h"
#include "sparse.h"
#include "module.h"
#include "optim.h"
#include "cuda_variable.h"

struct GCNParams {
    int num_nodes, input_dim, hidden_dim, output_dim;
    float dropout, learning_rate, weight_decay;
    int epochs, early_stopping;
    static GCNParams get_default();
};

class GCNData {
public:
    SparseIndex feature_index, graph;
    std::vector<int> split;
    std::vector<int> label;
    std::vector<float> feature_value;
};

class GCN {
    std::vector<Module*> modules;
    std::vector<Variable> variables;
    std::vector<CudaVariable> cuda_variables;
    Variable *input, *output;
    CudaVariable *cuda_input, *cuda_output;
    std::vector<int> truth;
    int *cuda_truth;
    Adam optimizer;
    float loss;
    float *cuda_loss;
    void set_input();
    void set_truth(int current_split);
    void set_cuda_input();
    float get_accuracy();
    float get_l2_penalty();
    std::pair<float, float> train_epoch();
    std::pair<float, float> eval(int current_split);
    GCNData *data;
public:
    GCN(GCNParams params, GCNData *data);
    GCN();
    GCNParams params;
    ~GCN();
    void run();
};

#define GCN_H
#endif
#include "../include/gcn.h"
#include "../include/timer.h"
#include "../include/cuda_check.h"
#include "../include/gpu_params.h"
#include <cstdio>
#include <tuple>

/**
 * Returns the default paramets of the model
 * they will be overwritten by the parser when reading the dataset
*/
GCNParams GCNParams::get_default() {
    /*
    return { // CITESEER
        3327,   // num_nodes
        3703,   // input_dim
        16,     // hidden_dim
        6,      // output_dim
        0.5,    // dropouyt
        0.01,   // learning_rate
        5e-4,   // weight_decay
        100,    // epochs
        0};     // early_stopping
    
    */

    ///*
    return { // CORA
        2708,   // num_nodes
        1433,   // input_dim
        16,     // hidden_dim
        7,      // output_dim
        0.5,    // dropouyt
        0.01,   // learning_rate
        5e-4,   // weight_decay
        100,    // epochs
        0};     // early_stopping
    //*/

    /*
    return { // PUBMED
        19717,   // num_nodes
        500,   // input_dim
        16,     // hidden_dim
        3,      // output_dim
        0.5,    // dropouyt
        0.01,   // learning_rate
        5e-4,   // weight_decay
        100,    // epochs
        0};     // early_stopping
    */

    /*
    return { // REDDIT
        232965,   // num_nodes
        602,   // input_dim
        128,     // hidden_dim
        41,      // output_dim
        0.5,    // dropouyt
        0.01,   // learning_rate
        5e-4,   // weight_decay
        100,    // epochs
        0};     // early_stopping
    */
}

GCN::GCN(GCNParams params, GCNData *input_data) {
    //init_rand_state();
    this->params = params;
    data = input_data;
    modules.reserve(8); // allocate the space for the 8 modules/layers
    cuda_variables.reserve(8);
    cuda_variables.emplace_back(data->feature_index.indices.size(), false);
    cuda_input = &cuda_variables.back();

    // dropout
    modules.push_back(new Dropout(cuda_input, params.dropout));
    cuda_variables.emplace_back(params.num_nodes * params.hidden_dim);
    CudaVariable *layer1_cuda_var1 = &cuda_variables.back();
    
    cuda_variables.emplace_back(params.input_dim * params.hidden_dim, true, true);
    CudaVariable *layer1_cuda_weight = &cuda_variables.back();
    layer1_cuda_weight->glorot(params.input_dim, params.hidden_dim); // weights initilization
    
    // sparsematmul
    modules.push_back(new SparseMatmul(cuda_input, layer1_cuda_weight, layer1_cuda_var1, &data->feature_index, params.num_nodes, params.input_dim, params.hidden_dim));
    cuda_variables.emplace_back(params.num_nodes * params.hidden_dim);
    CudaVariable *layer1_cuda_var2 = &cuda_variables.back();

    // graphsum
    modules.push_back(new GraphSum(layer1_cuda_var1, layer1_cuda_var2, &data->graph, params.hidden_dim));

    // ReLU
    modules.push_back(new ReLU(layer1_cuda_var2));

    // dropout
    modules.push_back(new Dropout(layer1_cuda_var2, params.dropout));
    cuda_variables.emplace_back(params.num_nodes * params.output_dim);
    CudaVariable *layer2_cuda_var1 = &cuda_variables.back();
    
    cuda_variables.emplace_back(params.hidden_dim * params.output_dim, true, true);
    CudaVariable *layer2_cuda_weight = &cuda_variables.back();
    layer2_cuda_weight->glorot(params.hidden_dim, params.output_dim); // weights initilization
    
    // matmul
    modules.push_back(new Matmul(layer1_cuda_var2, layer2_cuda_weight, layer2_cuda_var1, params.num_nodes, params.hidden_dim, params.output_dim));
    cuda_variables.emplace_back(params.num_nodes * params.output_dim);
    cuda_output = &cuda_variables.back();
    
    // graph sum
    modules.push_back(new GraphSum(layer2_cuda_var1, cuda_output, &data->graph, params.output_dim));
    check_call(cudaMalloc(&cuda_truth, params.num_nodes * sizeof(int)));
    
    // cross entropy loss
    check_call(cudaMalloc(&cuda_loss, sizeof(float)));
    modules.push_back(new CrossEntropyLoss(cuda_output, cuda_truth, &loss, cuda_loss, params.output_dim));

    // Adam optimization algorithm (alternative to the classical stochastic gradient descent)
    AdamParams adam_params = AdamParams::get_default();
    adam_params.lr = params.learning_rate;
    adam_params.weight_decay = params.weight_decay;
    optimizer = new Adam({{layer1_cuda_weight, true}, {layer2_cuda_weight, false}}, adam_params);
}

GCN::~GCN(){
    for(auto m: modules)
        delete m;
    delete optimizer;
    check_call(cudaFree(cuda_truth));
    check_call(cudaFree(cuda_loss));
}

// set the current input for the GCN model
void GCN::set_cuda_input() {
    check_call(cudaMemcpy(cuda_input->data, data->feature_value.data(), data->feature_value.size() * sizeof(float), cudaMemcpyHostToDevice));
}

// set the label of each node inside of the current_split (validation/train/test)
void GCN::set_cuda_truth(int current_split) {
    int *temp = (int *) malloc(params.num_nodes * sizeof(int));
    for(int i = 0; i < params.num_nodes; i++)
        // truth[i] is the real label of "i"
        temp[i] = data->split[i] == current_split ? data->label[i] : -1;
    check_call(cudaMemcpy(cuda_truth, temp, params.num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    free(temp);
}

__global__ void parallel_get_accuracy(int *wrong, int *total, int *truth, float *data, int N, int D) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N && truth[i] >= 0) {
        atomicAdd(total, 1);
        float truth_logit = data[i * D + truth[i]];
        for(int j = 0; j < D; j++)
            if (data[i * D + j] > truth_logit) {
                atomicAdd(wrong, 1);
                break;
            }
    }
}

// get the current accuracy of the model
float GCN::get_accuracy() {
    int wrong = 0, total = 0;
    int *cuda_wrong, *cuda_total;
    check_call(cudaMalloc(&cuda_wrong, sizeof(int)));
    check_call(cudaMalloc(&cuda_total, sizeof(int)));
    check_call(cudaMemcpy(cuda_wrong, &wrong, sizeof(int), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(cuda_total, &total, sizeof(int), cudaMemcpyHostToDevice));
    dim3 blocksPerGrid((params.num_nodes + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    parallel_get_accuracy<<<blocksPerGrid, threadsPerBlock>>>(cuda_wrong, cuda_total, cuda_truth, cuda_output->data, params.num_nodes, params.output_dim);
    //check_kernel_call();
    cudaDeviceSynchronize();
    check_call(cudaMemcpy(&wrong, cuda_wrong, sizeof(int), cudaMemcpyDeviceToHost));
    check_call(cudaMemcpy(&total, cuda_total, sizeof(int), cudaMemcpyDeviceToHost));
    check_call(cudaFree(cuda_wrong));
    check_call(cudaFree(cuda_total));
    return float(total - wrong) / total;
    /*
    int wrong = 0, total = 0;
    for(int i = 0; i < params.num_nodes; i++) {
        if(truth[i] < 0) continue;
        total++;
        float truth_logit = output->data[i * params.output_dim + truth[i]];
        for(int j = 0; j < params.output_dim; j++)
            if (output->data[i * params.output_dim + j] > truth_logit) {
                wrong++;
                break;
            }
    }
    return float(total - wrong) / total;
    */
}

__global__ void parallel_get_l2_penalty(float *l2, float *data, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float x = data[i];
        atomicAdd(l2, x * x);
    }
}

// reduce the likelihood of model overfitting by using l2 regularization
float GCN::get_l2_penalty() {
    float l2 = 0;
    float *cuda_l2;
    check_call(cudaMalloc(&cuda_l2, sizeof(float)));
    check_call(cudaMemcpy(cuda_l2, &l2, sizeof(float), cudaMemcpyHostToDevice));
    dim3 blocksPerGrid((cuda_variables[2].size + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    parallel_get_l2_penalty<<<blocksPerGrid, threadsPerBlock>>>(cuda_l2, cuda_variables[2].data, cuda_variables[2].size);
    //check_kernel_call();
    cudaDeviceSynchronize();
    check_call(cudaMemcpy(&l2, cuda_l2, sizeof(float), cudaMemcpyDeviceToHost));
    check_call(cudaFree(cuda_l2));
    return params.weight_decay * l2 / 2;
    /*
    float l2 = 0;
    for (int i = 0; i < variables[2].data.size(); i++) {
        float x = variables[2].data[i];
        l2 += x * x;
    }
    return params.weight_decay * l2 / 2;
    */
}

/**
 * Train an epoch of the model
*/
std::pair<float, float> GCN::train_epoch() {

    // Data transfer from host to device
    set_cuda_input(); // set the input data

    set_cuda_truth(1); // get the true labels for the dataset with split == 1 (train)

    for (auto m: modules) // iterate over the layer applying a forward pass
        m->forward(true);

    float train_loss = loss + get_l2_penalty(); // correct the loss with the l2 regularization
    float train_acc = get_accuracy(); // compute the accuracy comparing the prediction against the truth
    
    for (int i = modules.size() - 1; i >= 0; i--) // do a backward pass on the layers
        modules[i]->backward();

    optimizer->step(); // apply a step of the adapm optimization

    return {train_loss, train_acc};
}

/**
 * current_split == 2 --> validation
 * current_split == 3 --> test
*/
std::pair<float, float> GCN::eval(int current_split) {
    set_cuda_input();
    set_cuda_truth(current_split);
    for (auto m: modules)
        m->forward(false);
    float test_loss = loss + get_l2_penalty();
    float test_acc = get_accuracy();
    return {test_loss, test_acc};
}

void GCN::run() {
    int epoch = 1;
    std::vector<float> loss_history;
    // Iterate the training process based on the selected number of epoch
    for(; epoch <= params.epochs; epoch++) {
        float train_loss, train_acc, val_loss, val_acc;
        timer_start(TMR_TRAIN); // just for timing purposes
        std::tie(train_loss, train_acc) = train_epoch(); // train the epoch and record the current train_loss and train_accuracy
        std::tie(val_loss, val_acc) = eval(2); // eval the model at the current step in order to obtain the val_loss and val_accuracy
        printf("epoch=%d train_loss=%.5f train_acc=%.5f val_loss=%.5f val_acc=%.5f time=%.5f\n",
            epoch, train_loss, train_acc, val_loss, val_acc, timer_stop(TMR_TRAIN));

        loss_history.push_back(val_loss); // record the validation loss in order to apply an early stopping mechanism

        // early stopping mechanism
        if(params.early_stopping > 0 && epoch >= params.early_stopping) {
            float recent_loss = 0.0;
            for(int i = epoch - params.early_stopping; i < epoch; i++)
                recent_loss += loss_history[i];
            if (val_loss > recent_loss / params.early_stopping) {
                printf("Early stopping...\n");
                break;
            }
        }
    }
    PRINT_TIMER_AVERAGE(TMR_TRAIN, epoch);
    PRINT_TIMER_AVERAGE(TMR_MATMUL_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_MATMUL_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_SPMATMUL_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_SPMATMUL_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_GRAPHSUM_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_GRAPHSUM_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_RELU_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_RELU_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_DROPOUT_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_DROPOUT_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_LOSS_FW, epoch);
    

    float test_loss, test_acc;
    timer_start(TMR_TEST);
    std::tie(test_loss, test_acc) = eval(3); // eval the model on the test set
    printf("test_loss=%.5f test_acc=%.5f time=%.5f\n", test_loss, test_acc, timer_stop(TMR_TEST));
}

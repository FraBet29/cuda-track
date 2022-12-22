#include <iostream>
#include "../include/module.h"
#include "../include/optim.h"
#include "../include/parser.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

using namespace std;

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    if (argc < 2) {
        cout << "parallel_gcn graph_name [num_nodes input_dim hidden_dim output_dim"
                "dropout learning_rate, weight_decay epochs early_stopping]" << endl;
        return EXIT_FAILURE;
    }

    // Parse the selected dataset
    GCNParams params = GCNParams::get_default();
    GCNData data;
    std::string input_name(argv[1]);
    Parser parser(&params, &data, input_name);
    if (!parser.parse()) {
        std::cerr << "Cannot read input: " << input_name << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get gpu info
    int num_gpus;
    std::size_t free, total;
    cudaGetDeviceCount(&num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        cudaSetDevice(gpu_id);
        int id;
        cudaGetDevice(&id);
        cudaMemGetInfo(&free, &total);
        std::cout << "GPU " << id << " memory: free = " << free << ", total = " << total << std::endl;
    }

    GCN gcn(params, &data); // Create and initialize and object of type GCN.
    gcn.run(); // Run the main function of the model in order to train and validate the solution.

    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        cudaSetDevice(gpu_id);
        int id;
        cudaGetDevice(&id);
        cudaMemGetInfo(&free);
        std::cout << "GPU " << id << " memory: free = " << free << std::endl;
    }

    return EXIT_SUCCESS;
}
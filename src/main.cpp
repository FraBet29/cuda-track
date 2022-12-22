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
    int nDevices;
    cudaGetDeviceCount(&nDevices);
  
    std::cout << "Number of devices: " << nDevices << std::endl;
  
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device Number: " << i << std::endl;
        std::cout << "  Device name: " << prop.name << std::endl;
        std::cout << "  Global memory (GB): " << (float) prop.totalGlobalMem << std::endl;
        std::cout << "  Shared memory per block (KB) " << (float) prop.sharedMemPerBlock << std::endl;
        std::cout << "  Warp-size: " << prop.warpSize << std::endl;
        std::cout << "  Maximum number of threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Memory Clock Rate (MHz): " << prop.memoryClockRate << std::endl;
        std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
    }

    GCN gcn(params, &data); // Create and initialize and object of type GCN.
    gcn.run(); // Run the main function of the model in order to train and validate the solution.

    return EXIT_SUCCESS;
}
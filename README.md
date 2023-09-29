# GPU Acceleration track

In this folder, you will find a parallel GPU implementation of a Graph Convolutional Network in C++. This implementation accelerates a sequential CPU implementation following the original model developed by [Kipf et al.](https://arxiv.org/pdf/1609.02907.pdf). [Here](http://tkipf.github.io/graph-convolutional-networks/) you can find an additional blog post by the author of the paper detailing the model that was originally implemented with TensorFlow.

The goal of the project is to increase the performance of the CPU version by using a GPU as a hardware accelerator while maintaining an acceptable level of accuracy of the model compared to the baseline implementation and avoiding the usage of CUDA libraries (e.g., CuBLAS).

## Build and run
You can build and execute the code by running the following commands:

```sh
make
./exec/gcn-seq cora # dataset name from the dataset folder: [cora, pubmed, citeseer]
```

## Project structure
In the `.\src\` folder you will find the main components of the implementation.
As usual, the core is the `main.cpp` file that parses the selected dataset and creates an object of type `GCN`.
During the initialization phase, all the layers of the model are constructed. 
Consequently, the model is then run by calling the function `GCN::run()`.
This function, based on the parameters set by `GCN::get_default()`, will execute a predefined number of epochs during the training phase.

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

// Utility function to read data from a file into a vector
std::vector<float> readDataFromFile(const std::string &filename) {
    std::vector<float> data;
    std::ifstream file(filename);
    float value;
    while (file >> value) {
        data.push_back(value);
    }
    return data;
}

class FH {
private:
    std::ifstream _file;
public:
    FH(const std::string &filename) {
        _file = std::ifstream(filename);
    }

    std::vector<float> readDataFromFile() {
        std::vector<float> data;
        float value;
        while (_file >> value) {
            data.push_back(value);
        }
        return data;
        }
};

__global__ void addArrays(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Main function
int main(int argc, char *argv[]) {
    // Check for correct argument count
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <file1> <file2> <output_file>" << std::endl;
        return 1;
    }

    // Read data from files
    FH host_A_file(argv[1]);
    FH host_B_file(argv[2]);

    std::vector<float> host_A = host_A_file.readDataFromFile();
    std::vector<float> host_B = host_B_file.readDataFromFile();
    int numElements = host_A.size();

    // Allocate memory on the GPU
    float *device_A, *device_B, *device_C;
    cudaMalloc((void **)&device_A, numElements * sizeof(float));
    cudaMalloc((void **)&device_B, numElements * sizeof(float));
    cudaMalloc((void **)&device_C, numElements * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(device_A, host_A.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, numElements);

    // Copy result back to host
    std::vector<float> host_C(numElements);
    cudaMemcpy(host_C.data(), device_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Write result to file
    std::ofstream outputFile(argv[3]);
    for (float value : host_C) {
        outputFile << value << std::endl;
    }

    // Free device memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    return 0;
}

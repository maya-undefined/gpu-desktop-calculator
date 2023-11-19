#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

#define _CHUNK_SIZE 1 * 1024 * 1024
#define NUM_STREAMS 5

class FH {
private:
    std::ifstream _file;
    const uint _chunksize = _CHUNK_SIZE;
public:
    FH(const std::string &filename) {
        _file = std::ifstream(filename);
    }

    bool eof() {
        return _file.eof();
    }

    std::vector<float> readDataFromFile() {
        std::vector<float> data;
        float value;
        while (data.size() < _chunksize) {
            if (!(_file >>value)) break;
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

    // Write result to file
    std::ofstream outputFile(argv[3]);


    // Allocate memory on the GPU
    float *device_A, *device_B, *device_C;
    cudaMalloc((void **)&device_A, _CHUNK_SIZE * sizeof(float));
    cudaMalloc((void **)&device_B, _CHUNK_SIZE * sizeof(float));
    cudaMalloc((void **)&device_C, _CHUNK_SIZE * sizeof(float));

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    while (!host_A_file.eof()) {
        std::vector<float> host_A_chunk[NUM_STREAMS];
        std::vector<float> host_B_chunk[NUM_STREAMS];
        std::vector<float> host_C_chunk[NUM_STREAMS];

        // std::vector<float> host_A = host_A_file.readDataFromFile();
        // std::vector<float> host_B = host_B_file.readDataFromFile();
        for (int i = 0; i < NUM_STREAMS; ++i) {
            host_A_chunk[i] = host_A_file.readDataFromFile();
            host_B_chunk[i] = host_B_file.readDataFromFile();
        }

        int numElements = host_A_chunk[0].size();

        for (int i = 0; i < NUM_STREAMS; ++i) {

            // Copy data from host to device
            // // cudaMemcpy(device_A, host_A.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
            // // cudaMemcpy(device_B, host_B.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
            // cudaMemcpyAsync(device_A, host_A.data(), numElements * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
            // cudaMemcpyAsync(device_B, host_B.data(), numElements * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(device_A, host_A_chunk[i].data(), _CHUNK_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(device_B, host_B_chunk[i].data(), _CHUNK_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

            // Launch the CUDA Kernel
            int threadsPerBlock = 256;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
            // addArrays<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, numElements);
            addArrays<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(device_A, device_B, device_C, _CHUNK_SIZE);

            // Copy result back to host
            // std::vector<float> host_C(numElements);
            // cudaMemcpy(host_C.data(), device_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);
            host_C_chunk[i].resize(numElements);
            cudaMemcpyAsync(host_C_chunk[i].data(), device_C, _CHUNK_SIZE * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        } // for each stream

        // Synchronize streams
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamSynchronize(streams[i]);
        }

        for (int i = 0; i < NUM_STREAMS; i ++) {
            for (float value : host_C_chunk[i]) {
                outputFile << std::fixed << std::setprecision(6) << value << std::endl;
            }            
        }
    } // while data exists

    // Free device memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    return 0;
}

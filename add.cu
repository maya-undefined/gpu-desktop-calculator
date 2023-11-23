#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

#define _CHUNK_SIZE 1024 * 1024 * 50

class FH {
private:
    std::ifstream _file;
    const uint _chunksize = _CHUNK_SIZE;
public:
    FH(const std::string &filename) {
        _file = std::ifstream(filename);
        char* buffer = new char[_CHUNK_SIZE];
        _file.rdbuf()->pubsetbuf(buffer, _CHUNK_SIZE);
    }

    bool eof() {
        return _file.eof();
    }
    std::vector<float> parse_line_of_floats(const std::string& line) {
        std::vector<float> numbers;
        const char* str = line.c_str();
        char* end;

        while (true) {
            float num = std::strtof(str, &end);
            if ( end == str) break;
            numbers.push_back(num);
            str = end;
        }

        return numbers;
    }
    static size_t total_vector_size(std::vector<std::vector<float> > data) {
        size_t total_size = 0;
        for (std::vector<float> _d : data) {
            total_size += _d.size();
        }
        return total_size;
    }

    std::vector<float> readDataFromFile() {
        std::vector<std::vector<float> > data;
        std::string line;

        size_t total_size = 0;

        while (total_size < _chunksize) {
            if (!(std::getline(_file, line))) break;

            std::vector<float> numbers = parse_line_of_floats(line);

            if (data.size() == 0) {
                data.resize(numbers.size());
            }

            for (int i = 0; i < numbers.size(); i++) {
                data[i].push_back(numbers[i]);
            }

            total_size = total_vector_size(data);
        }

        return data[0];
    }
};

__global__ void addArrays(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

__global__ void complexPhysicsCalculation(const float *A, const float *B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        // Hypothetical physics-based calculation
        float temp = A[i] * expf(-B[i] / A[i]);
        float result = sinf(A[i]) * cosf(B[i]) + temp;

        // Store the result
        C [i]= result;
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

    char* buffer = new char[_CHUNK_SIZE];
    outputFile.rdbuf()->pubsetbuf(buffer, _CHUNK_SIZE);

    // Allocate memory on the GPU
    float *device_A, *device_B, *device_C;
    cudaMalloc((void **)&device_A, _CHUNK_SIZE * sizeof(float));
    cudaMalloc((void **)&device_B, _CHUNK_SIZE * sizeof(float));
    cudaMalloc((void **)&device_C, _CHUNK_SIZE * sizeof(float));

    while (!host_A_file.eof()) {
        std::vector<float> host_A = host_A_file.readDataFromFile();
        std::vector<float> host_B = host_B_file.readDataFromFile();
        int numElements = host_A.size();

        // Copy data from host to device
        cudaMemcpy(device_A, host_A.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_B, host_B.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);

        // Launch the CUDA Kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        addArrays<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, numElements);
        // complexPhysicsCalculation<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, numElements);

        // Copy result back to host
        std::vector<float> host_C(numElements);
        cudaMemcpy(host_C.data(), device_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

        for (float value : host_C) {
            outputFile << value << "\n";
        }

    }

    // Free device memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    return 0;
}

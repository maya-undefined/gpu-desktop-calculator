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
        char* end = NULL;

        while (true) {
            float num = std::strtof(str, &end);
            // is this safe?

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

    std::vector<std::vector<float> > readDataFromFile() {
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

        return data;
    }
};

__global__ void addMultipleArrays(float *A, float *B, float *C, int A_rows, int B_rows, int A_cols, int B_cols) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < A_rows) {
        float sum = 0;
        int col, idx;
        for ( col = 0; col < A_cols; ++col) {
            idx = row * A_cols + col;
            sum += A[idx];
        }

        for ( col = 0; col < B_cols; ++col) {
            idx = row * B_cols + col;
            sum += B[idx];
        }

        
        C[row] = sum;
    }
}

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
        C[i]= result;
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

    while (!host_A_file.eof()) {
        std::vector<std::vector<float> > host_A = host_A_file.readDataFromFile();
        std::vector<std::vector<float> > host_B = host_B_file.readDataFromFile();
        int numElements = host_A.size();

        // Allocate memory on the GPU
        float *device_C;
        float *device_A, *device_B;

        cudaMalloc((void **)&device_C, _CHUNK_SIZE * sizeof(float));

        std::vector<float> flat_A;
        std::vector<float> flat_B;
        for (const auto& row: host_A) {
            flat_A.insert(flat_A.end(), row.begin(), row.end());
        }

        for (const auto& row: host_B) {
            flat_B.insert(flat_B.end(), row.begin(), row.end());
        }
        
        cudaMalloc(&device_A, flat_A.size() * sizeof(float));
        cudaMalloc(&device_B, flat_B.size() * sizeof(float));
        cudaMemcpy(device_A, flat_A.data(), flat_A.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_B, flat_B.data(), flat_B.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Launch the CUDA Kernel
        dim3 blockSize(256);
        dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);
        addMultipleArrays<<<gridSize, blockSize>>>(
                device_A, device_B, device_C, 
                host_A[0].size(), host_B[0].size(),
                host_A.size(), host_B.size() // rows
                ); // columns

        // Copy result back to host
        std::vector<float> host_C(host_A[0].size());
        cudaMemcpy(host_C.data(), device_C, host_A.size() * sizeof(float), cudaMemcpyDeviceToHost);

        for (float value : host_C) {
            outputFile << value << "\n";
        }

        cudaFree(device_A);
        cudaFree(device_B);
        cudaFree(device_C);
    }



    return 0;
}

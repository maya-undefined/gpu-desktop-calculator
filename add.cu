#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

#define _CHUNK_SIZE 1024 * 64 * 1024
// #define _CHUNK_SIZE 1

class FH {
private:
    std::ifstream _file;
    std::string _file_name;
    const uint _chunksize = _CHUNK_SIZE;
    size_t _row_len;
    size_t _col_len;
public:
    FH(const std::string &filename) {
        _file_name = filename;
        _file = std::ifstream(filename);
        char* buffer = new char[_CHUNK_SIZE];
        _file.rdbuf()->pubsetbuf(buffer, _CHUNK_SIZE);
        _row_len = 0;
        _col_len = 0;
    }

    bool eof() {
        return _file.eof();
    }

    size_t col_len() {
        return _col_len;
    }

    size_t row_len() {
        // only call this after calling read_data_from_file >_< sry
        return _row_len;
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

    std::vector<float> read_data_from_file() {
        // GPUs like contigious, flat lengths of memory

        std::string line;
        std::vector<float> data;

        size_t total_size = 0;
        while (total_size < _chunksize) {
            if (!(std::getline(_file, line))) break;

            std::vector<float> numbers = parse_line_of_floats(line);
            if (_col_len == 0) {
                _col_len = numbers.size();
            }

            data.insert(data.end(), numbers.begin(), numbers.end());
            _row_len += 1;
            total_size += numbers.size() * sizeof(float);
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
    float *device_C;
    float *device_A, *device_B;

    size_t A_rows, A_cols, B_rows, B_cols;
    A_rows = 0; A_cols = 1; B_rows = 0; B_cols = 1;
    size_t loops = 0;
    while (!host_A_file.eof()) {
        A_rows = host_A_file.row_len();
        B_rows = host_B_file.row_len();
        // Remember how many rows we read so far

        std::vector<float> host_A = host_A_file.read_data_from_file();
        std::vector<float> host_B = host_B_file.read_data_from_file();

        A_rows = host_A_file.row_len() - A_rows;
        B_rows = host_B_file.row_len() - B_rows;
        // and now we can calculate how many rows we need to process in this chunk

        if (B_cols != host_B_file.col_len()) { B_cols = host_B_file.col_len(); }
        if (A_cols != host_A_file.col_len()) { A_cols = host_A_file.col_len(); }

        cudaMalloc((void **)&device_C, host_A.size() * sizeof(float));
        cudaMalloc((void **)&device_A, host_A.size() * sizeof(float));
        cudaMalloc((void **)&device_B, host_B.size() * sizeof(float));
        cudaMemcpy(device_A, host_A.data(), host_A.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_B, host_B.data(), host_B.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Launch the CUDA Kernel
        dim3 blockSize(768);
        dim3 gridSize((A_rows + blockSize.x - 1) / blockSize.x);
        addMultipleArrays<<<gridSize, blockSize>>>(
                device_A, device_B, device_C, 
                A_rows, B_rows, // rows
                A_cols, B_cols // columns
                ); 

        // Copy result back to host
        // we only need to keep track of how many elements since we are using a flat array
        size_t ele_to_read = max(A_rows, B_rows);
        std::vector<float> host_C(ele_to_read);
        cudaMemcpy(host_C.data(), device_C, ele_to_read * sizeof(float), cudaMemcpyDeviceToHost);

        for (float value : host_C) {
            outputFile << std::fixed << std::setprecision(6) << value << "\n";
        }
        loops++;
        cudaFree(device_A);
        cudaFree(device_B);
        cudaFree(device_C);
    }

    return 0;
}

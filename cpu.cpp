#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>


#define _CHUNK_SIZE 5 * 1024 * 1024
#define NUM_STREAMS 1

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
    std::ofstream outputFile(argv[3]);
    char* buffer = new char[_CHUNK_SIZE];
    outputFile.rdbuf()->pubsetbuf(buffer, _CHUNK_SIZE);

    while (!host_A_file.eof()) {
        std::vector<float> host_A = host_A_file.readDataFromFile();
        std::vector<float> host_B = host_B_file.readDataFromFile();

        for (int i = 0; i < host_A.size(); i++) {
            float value = host_A[i] + host_B[i];
            // float temp = host_A[i] * std::exp(-host_B[i] / host_A[i]);
            // float value = std::sin(host_A[i]) * std::cos(host_B[i]) + temp;

            // Store the result
            // C [i]= result;
            outputFile << std::fixed << std::setprecision(6) << value << "\n";
        }
    }
}
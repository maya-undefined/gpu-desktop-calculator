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
    std::vector<std::vector<float> > readDataFromFile() {
        std::vector<std::vector<float> > data;
        std::string line;
        float value;
        while (data.size() < _chunksize) {
            if (!(std::getline(_file, line))) break;

            std::vector<float> numbers = parse_line_of_floats(line);
            data.push_back(numbers);
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
        std::vector<std::vector<float> > host_A = host_A_file.readDataFromFile();
        std::vector<std::vector<float> > host_B = host_B_file.readDataFromFile();

        for (int i = 0; i < host_A.size(); i++) {
            float sum = 0;
            for (int j = 0; j < host_A[i].size(); j++) {
                sum += host_A[i][j];
            }

            for (int j = 0; j < host_B[i].size(); j++) {
                sum += host_B[i][j];
            }

            // float value = host_A[i] + host_B[i];
            // float temp = host_A[i] * std::exp(-host_B[i] / host_A[i]);
            // float value = std::sin(host_A[i]) * std::cos(host_B[i]) + temp;

            // Store the result
            // C [i]= result;
            outputFile << std::fixed << std::setprecision(6) << sum << "\n";
        }
    }
}
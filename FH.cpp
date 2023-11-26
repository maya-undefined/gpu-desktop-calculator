#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include "FH.h"
#include "gdc.h"

 FH::FH(const std::string &filename) {
        _file_name = filename;
        _file = std::ifstream(filename);
        char* buffer = new char[_CHUNK_SIZE];
        _file.rdbuf()->pubsetbuf(buffer, _CHUNK_SIZE);
        _row_len = 0;
        _col_len = 0;
    }

std::vector<float> FH::parse_line_of_floats(const std::string& line) {
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

size_t FH::total_vector_size(std::vector<std::vector<float> > data) {
    size_t total_size = 0;
    for (std::vector<float> _d : data) {
        total_size += _d.size();
    }
    return total_size;
}

std::vector<float> FH::read_data_from_file() {
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

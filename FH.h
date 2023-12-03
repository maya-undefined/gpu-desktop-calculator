#ifndef _FH_H
#define _FH_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include "gdc.h"

class FH {
private:
    std::ifstream _file;
    std::string _file_name;
    const uint _chunksize = _CHUNK_SIZE;
    size_t _row_len;
    size_t _col_len;
public:
    FH(const std::string &filename);
    bool eof() { return _file.eof(); }
    size_t col_len() { return _col_len; }
    size_t row_len() { return _row_len; }
    std::vector<float> parse_line_of_floats(const std::string& line);
    static size_t total_vector_size(std::vector<std::vector<float> > data);
    std::vector<float> read_data_from_file();
};

#endif
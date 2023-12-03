#include "verb.h"
#include "kernels.h"

Verb::Verb(std::string file1, std::string outputFile) {
       host_A_file = new FH(file1);
       outputFilePtr = new std::ofstream(outputFile);

}

Verb::Verb(std::string file1, std::string file2, std::string outputFile) {
       host_A_file = new FH(file1);
       host_B_file = new FH(file2);
       outputFilePtr =  new std::ofstream(outputFile);
}

void Verb::dispatch() {
       if (host_B_file == nullptr) {
              // 1 file operations
              std::cout << "verb::dispatch 1 file" << std::endl;
              execute();
       } else {

              size_t loops = 0;
              while (!host_A_file->eof()) {
                     A_rows = host_A_file->row_len();
                     B_rows = host_B_file->row_len();
                     // Remember how many rows we read so far

                     std::vector<float> host_A = host_A_file->read_data_from_file();
                     std::vector<float> host_B = host_B_file->read_data_from_file();

                     A_rows = host_A_file->row_len() - A_rows;
                     B_rows = host_B_file->row_len() - B_rows;
                     // and now we can calculate how many rows we need to process in this chunk

                     if (B_cols != host_B_file->col_len()) { B_cols = host_B_file->col_len(); }
                     if (A_cols != host_A_file->col_len()) { A_cols = host_A_file->col_len(); }

                     cudaMalloc((void **)&device_C, host_A.size() * sizeof(float));
                     cudaMalloc((void **)&device_A, host_A.size() * sizeof(float));
                     cudaMalloc((void **)&device_B, host_B.size() * sizeof(float));
                     cudaMemcpy(device_A, host_A.data(), host_A.size() * sizeof(float), cudaMemcpyHostToDevice);
                     cudaMemcpy(device_B, host_B.data(), host_B.size() * sizeof(float), cudaMemcpyHostToDevice);

                     execute();

                     // Copy result back to host
                     // we only need to keep track of how many elements since we are using a flat array
                     size_t ele_to_read = max(A_rows, B_rows);
                     std::vector<float> host_C(ele_to_read);
                     cudaMemcpy(host_C.data(), device_C, ele_to_read * sizeof(float), cudaMemcpyDeviceToHost);

                     for (float value : host_C) {
                            *outputFilePtr << std::fixed << std::setprecision(6) << value << "\n";
                     }
                     outputFilePtr->flush();
                     loops++;
                     cudaFree(device_A);
                     cudaFree(device_B);
                     cudaFree(device_C);
              } // while
       } // if 1 file or 2 file operations
}

void Add::execute() {
       dim3 blockSize(256);
       dim3 gridSize((A_rows + blockSize.x - 1) / blockSize.x);
       addMultipleArrays<<<gridSize, blockSize>>>(
              device_A, device_B, device_C, 
              A_rows, B_rows, // rows
              A_cols, B_cols // columns
       ); 

}

void Exp::execute() {
       std::cout << "Exp" << std::endl;
}
#include "gdc.h"
#include "FH.h"
#include "verb.h"

// Main function
int main(int argc, char *argv[]) {
    // Check for correct argument count
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <verb> <file1> <file2> <output_file>" << std::endl;
        std::cerr << "Verb: (add|mul|div|exp)" << std::endl; 
        return 1;
    }

    std::string verb = std::string(argv[1]);
    if (verb != "mul" && verb != "add" && verb != "div" && verb != "exp") {
        std::cerr << "Verb: (add|mul|div|exp)" << std::endl; 
        std::cerr << "Invalid verbs" << std::endl;
    }

    // Write result to file
    std::ofstream outputFile(argv[4]);

    char* buffer = new char[_CHUNK_SIZE];
    outputFile.rdbuf()->pubsetbuf(buffer, _CHUNK_SIZE);
    // Allocate memory on the GPU

    // Read data from files
    FH host_A_file(argv[2]);
    FH host_B_file(argv[3]);

    Add add(argv[2], argv[3], argv[4]);
    add.dispatch();

    // Exp exp(argv[2]);
    // exp.dispatch();

    // size_t A_rows, A_cols, B_rows, B_cols;
    // A_rows = 0; A_cols = 1; B_rows = 0; B_cols = 1;
    // size_t loops = 0;
    // while (!host_A_file.eof()) {
    //     A_rows = host_A_file.row_len();
    //     B_rows = host_B_file.row_len();
    //     // Remember how many rows we read so far

    //     std::vector<float> host_A = host_A_file.read_data_from_file();
    //     std::vector<float> host_B = host_B_file.read_data_from_file();

    //     A_rows = host_A_file.row_len() - A_rows;
    //     B_rows = host_B_file.row_len() - B_rows;
    //     // and now we can calculate how many rows we need to process in this chunk

    //     if (B_cols != host_B_file.col_len()) { B_cols = host_B_file.col_len(); }
    //     if (A_cols != host_A_file.col_len()) { A_cols = host_A_file.col_len(); }

    //     cudaMalloc((void **)&device_C, host_A.size() * sizeof(float));
    //     cudaMalloc((void **)&device_A, host_A.size() * sizeof(float));
    //     cudaMalloc((void **)&device_B, host_B.size() * sizeof(float));
    //     cudaMemcpy(device_A, host_A.data(), host_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    //     cudaMemcpy(device_B, host_B.data(), host_B.size() * sizeof(float), cudaMemcpyHostToDevice);

    //     // Launch the CUDA Kernel
    //     dim3 blockSize(256);
    //     dim3 gridSize((A_rows + blockSize.x - 1) / blockSize.x);
    //     if (verb == "add") {
    //         addMultipleArrays<<<gridSize, blockSize>>>(
    //                 device_A, device_B, device_C, 
    //                 A_rows, B_rows, // rows
    //                 A_cols, B_cols // columns
    //                 ); 
    //     } else
    //     if (verb == "mul") {
    //         mulMultipleArrays<<<gridSize, blockSize>>>(
    //             device_A, device_B, device_C, 
    //             A_rows, B_rows, // rows
    //             A_cols, B_cols // columns
    //             ); 
    //     } else
    //     if (verb == "div") {
    //         divArrays<<<gridSize, blockSize>>>(
    //             device_A, device_B, device_C,
    //             A_rows, B_rows
    //         );
    //     } else
    //     if (verb == "exp") {
    //         expArrays<<<gridSize, blockSize>>>(
    //             device_A, device_C,
    //             A_rows
    //         );
    //     }

    //     // Copy result back to host
    //     // we only need to keep track of how many elements since we are using a flat array
    //     size_t ele_to_read = max(A_rows, B_rows);
    //     std::vector<float> host_C(ele_to_read);
    //     cudaMemcpy(host_C.data(), device_C, ele_to_read * sizeof(float), cudaMemcpyDeviceToHost);

    //     for (float value : host_C) {
    //         outputFile << std::fixed << std::setprecision(6) << value << "\n";
    //     }
    //     loops++;
    //     cudaFree(device_A);
    //     cudaFree(device_B);
    //     cudaFree(device_C);
    // }

    return 0;
}

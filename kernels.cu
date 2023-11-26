#include "kernels.h"
#include "gdc.h"

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
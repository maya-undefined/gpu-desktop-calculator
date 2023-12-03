#include "gdc.h"

__global__ void addMultipleArrays(float *A, float *B, float *C, int A_rows, int B_rows, int A_cols, int B_cols);
__global__ void mulMultipleArrays(float *A, float *B, float *C, int A_rows, int B_rows, int A_cols, int B_cols);

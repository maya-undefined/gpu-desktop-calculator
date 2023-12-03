#ifndef _KERNELS_H
#define _KERNELS_H

__global__ void addMultipleArrays(const float *A, const float *B, float *C, int A_rows, int B_rows, int A_cols, int B_cols);
__global__ void mulMultipleArrays(const float *A, const float *B, float *C, int A_rows, int B_rows, int A_cols, int B_cols);
__global__ void divArrays(const float *A, const float *B, float *C, int A_rows, int B_rows);
__global__ void expArrays(const float *A, float *C, int A_rows);

#endif
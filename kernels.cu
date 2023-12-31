#include "kernels.h"
#include "gdc.h"

__global__ void addMultipleArrays(const float *A, const float *B, float *C, int A_rows, int B_rows, int A_cols, int B_cols) {
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

__global__ void mulMultipleArrays(const float *A, const float *B, float *C, int A_rows, int B_rows, int A_cols, int B_cols) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < A_rows) {
        float sum = 1;
        int col, idx;
        for ( col = 0; col < A_cols; ++col) {
            idx = row * A_cols + col;
            sum *= A[idx];
        }

        for ( col = 0; col < B_cols; ++col) {
            idx = row * B_cols + col;
            sum *= B[idx];
        }

        
        C[row] = sum;
    }
}

__global__ void divArrays(const float *A, const float *B, float *C, int A_rows, int B_rows) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < A_rows) {
        C[i] = A[i] / B[i];
    }
}

__global__ void addArrays(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

__global__ void expArrays(const float *A, float *C, int A_rows) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < A_rows) {
        C[i] = expf(A[i]);
    }
}

__global__ void sinArrays(const float *A, float *C, int A_rows) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < A_rows) {
        C[i] = sinf(A[i]);
    }
}
__global__ void cosArrays(const float *A, float *C, int A_rows) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < A_rows) {
        C[i] = cosf(A[i]);
    }
}

__global__ void tanArrays(const float *A, float *C, int A_rows) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < A_rows) {
        C[i] = tanf(A[i]);
    }
}

__global__ void complexPhysicsCalculation(const float *A, const float *B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        // Hypothetical physics-based calculation
        float temp = A[i] * expf(-B[i] / A[i]);
        float result = sinf(A[i]) * cosf(B[i]) + temp;

        // Store the result
        C[i]= result;
    }
}

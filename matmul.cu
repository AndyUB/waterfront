#include <iostream>
#include <cuda_runtime.h>

// Function to check CUDA errors
#define CUDA_CHECK_ERROR(err) do { cudaError_t err__ = (err); if (err__ != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err__) << " at line " << __LINE__ << std::endl; exit(1); \
} } while(0)

__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

int main() {
    const int N = 1000;
    size_t size = N * N * sizeof(float);

    // Allocate memory on the GPU
    float *d_A, *d_B, *d_C;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_C, size));

    // Allocate memory on the host
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices A and B on the host
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Copy matrices A and B from host to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Perform matrix multiplication on the GPU
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Copy result matrix C from device to host
    CUDA_CHECK_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Print a part of the result matrix
    std::cout << "C[0]: " << h_C[0] << std::endl;

    // Free GPU memory
    CUDA_CHECK_ERROR(cudaFree(d_A));
    CUDA_CHECK_ERROR(cudaFree(d_B));
    CUDA_CHECK_ERROR(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

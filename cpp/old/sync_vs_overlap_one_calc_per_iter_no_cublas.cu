#include <iostream>
#include <cublas_v2.h>

inline void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void checkCublasError(cublasStatus_t err) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << err << " at line " << __LINE__ << std::endl;
        exit(1);
    }
}

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

void maxError(float *output, int iterations, int N, bool isC) {
    float maxErr = 0;
    int maxI = -1, maxJ = -1;
    const float epsilon = 1e-6;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            float diff = std::abs(output[i * N + j]);
            if (diff > maxErr) {
                maxErr = diff;
                maxI = i;
                maxJ = j;
            }
            if (diff > epsilon) {
                std::cerr << "Error: " << (isC ? "C" : "F") << ": " << i << ", " << j << " (" << output[i * N + j] << ")!" << std::endl;
            }
        }
        {
            float diff = std::abs(std::pow((float) i / N, isC ? 2 : 3) - output[i * N + i]);
            if (diff > maxErr) {
                maxErr = diff;
                maxI = i;
                maxJ = i;
            }
            if (diff > epsilon) {
                std::cerr << "Error: " << (isC ? "C" : "F") << ": " << i << ", " << i << " (" << output[i * N + i] << ")!" << std::endl;
            }
        }
        for (int j = i + 1; j < N; ++j) {
            float diff = std::abs(output[i * N + j]);
            if (diff > maxErr) {
                maxErr = diff;
                maxI = i;
                maxJ = j;
            }
            if (diff > epsilon) {
                std::cerr << "Error: " << (isC ? "C" : "F") << ": " << i << ", " << j << " (" << output[i * N + j] << ")!" << std::endl;
            }
        }
    }
    std::cout << "Max error: " << maxErr << std::endl;
    if (maxI != -1) {
        std::cout << "at " << maxI << ", " << maxJ << ": " << output[maxI * N + maxJ] << std::endl;
    }
}

float experiment(bool overlap, int N, int iterations) {
    size_t size = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    checkCudaError(cudaSetDevice(0));
    checkCudaError(cudaMalloc((void**)&d_A, size));
    checkCudaError(cudaMalloc((void**)&d_B, size));
    checkCudaError(cudaMalloc((void**)&d_C, size));

    float *d_D, *d_E, *d_F;
    checkCudaError(cudaSetDevice(1));
    checkCudaError(cudaMalloc((void**)&d_D, size));
    checkCudaError(cudaMalloc((void**)&d_E, size));
    checkCudaError(cudaMalloc((void**)&d_F, size));

    float *h, *h_C, *h_F;
    checkCudaError(cudaMallocHost((void**)&h, size));
    checkCudaError(cudaMallocHost((void**)&h_C, size));
    checkCudaError(cudaMallocHost((void**)&h_F, size));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (j == i) {
                h[i * N + i] = (float) i / N;
            } else {
                h[i * N + j] = 0;
            }
        }
    }

    float elapse;

    cudaEvent_t startEvent, endEvent;
    checkCudaError(cudaEventCreate(&startEvent));
    checkCudaError(cudaEventCreate(&endEvent));

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream));

    checkCudaError(cudaMemcpy(d_A, h, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, h, size, cudaMemcpyHostToDevice));

    checkCudaError(cudaSetDevice(1));
    checkCudaError(cudaMemcpy(d_D, h, size, cudaMemcpyHostToDevice));

    checkCudaError(cudaEventRecord(startEvent, 0));
    checkCudaError(cudaEventSynchronize(startEvent));
    checkCudaError(cudaEventRecord(startEvent, 0));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    for (int i = 0; i < iterations; ++i) {
        checkCudaError(cudaSetDevice(0));
        matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        // checkCublasError(cublasSgemm(handle0, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));
        // heavier computation leads to higher speedup
        if (overlap) {
            checkCudaError(cudaMemcpyPeerAsync(d_E, 1, d_C, 0, size, stream));
        } else {
            checkCudaError(cudaMemcpyPeer(d_E, 1, d_C, 0, size));
            // checkCudaError(cudaStreamSynchronize(0));
        }
        checkCudaError(cudaSetDevice(1));
        matmul<<<gridDim, blockDim>>>(d_D, d_E, d_F, N);
        // checkCublasError(cublasSgemm(handle1, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_D, N, d_E, N, &beta, d_F, N));
    }

    checkCudaError(cudaEventRecord(endEvent, 0));
    checkCudaError(cudaEventSynchronize(endEvent));
    checkCudaError(cudaEventElapsedTime(&elapse, startEvent, endEvent));

    std::cout << "Total time (" << (overlap ? "Overlapping" : "Synchronous") << "): " << elapse << " ms\n";

    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(h_F, d_F, size, cudaMemcpyDeviceToHost));

    maxError(h_C, iterations, N, true);
    maxError(h_F, iterations, N, false);

    checkCudaError(cudaSetDevice(0));
    checkCudaError(cudaFree(d_A));
    checkCudaError(cudaFree(d_B));
    checkCudaError(cudaFree(d_C));

    checkCudaError(cudaSetDevice(1));
    checkCudaError(cudaFree(d_D));
    checkCudaError(cudaFree(d_E));
    checkCudaError(cudaFree(d_F));

    checkCudaError(cudaEventDestroy(startEvent));
    checkCudaError(cudaEventDestroy(endEvent));

    checkCudaError(cudaStreamDestroy(stream));

    checkCudaError(cudaFreeHost(h));
    checkCudaError(cudaFreeHost(h_C));
    checkCudaError(cudaFreeHost(h_F));

    return elapse;
}

int main() {
    const int experiments = 1;
    const int Ns[experiments] = {1000};
    // const int experiments = 11;
    // for N in [417, 1499], CUBLAS_STATUS_EXECUTION_FAILED error???
    // const int Ns[experiments] = {400, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
    // const int Ns[experiments] = {400, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000};
    const int iterations = 10;

    for (int i = 0; i < experiments; i++) {
        const int N = Ns[i];
        std::cout << "=============================="
                << std::endl << "N = " << N << std::endl;
        std::cout << "Running synchronous version...\n";
        float sync = experiment(false, N, iterations);

        std::cout << "Running overlapping version...\n";
        float async = experiment(true, N, iterations);

        float speedup = (sync - async) / (sync);
        std::cout << "speedup: " << speedup << std::endl;
    }

    return 0;
}

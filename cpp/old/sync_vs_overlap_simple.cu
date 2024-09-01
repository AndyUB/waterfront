#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

#define CUDA_CHECK_ERROR(err) do { cudaError_t err__ = (err); if (err__ != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err__) << " at line " << __LINE__ << std::endl; exit(1); \
} } while(0)

#define CUBLAS_CHECK_ERROR(err) do { cublasStatus_t err__ = (err); if (err__ != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLAS Error: " << err__ << " at line " << __LINE__ << std::endl; exit(1); \
} } while(0)

std::chrono::nanoseconds runExperiment(bool overlap, int N, int iterations) {
    size_t size = N * N * sizeof(float);
    // std::cout << size << std::endl;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocate memory on GPU0
    float *d_A, *d_B, *d_C;
    CUDA_CHECK_ERROR(cudaSetDevice(0));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_C, size));

    // Allocate memory on GPU1
    float *d_D, *d_E;
    CUDA_CHECK_ERROR(cudaSetDevice(1));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_D, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_E, size));

    // Initialize matrices A, B, D on the host
    // float *h_A = (float*)calloc(N * N, sizeof(float));
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_D = (float*)malloc(size);
    // float *h_D = (float*)calloc(N * N, sizeof(float));
    // for (int i = 0; i < N; ++i) {
    //     h_A[i * N + i] = (float) i / N;
    //     h_B[i * N + i] = (float) i / N;
    //     h_D[i * N + i] = (float) i / N;
    // }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (j == i) {
                h_A[i * N + i] = (float) i / N;
                h_B[i * N + i] = (float) i / N;
                h_D[i * N + i] = (float) i / N;
            } else {
                h_A[i * N + j] = 0;
                h_B[i * N + j] = 0;
                h_D[i * N + j] = 0;
            }
        }
    }

    // Copy matrices A, B, D from host to device
    CUDA_CHECK_ERROR(cudaSetDevice(0));
    CUDA_CHECK_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    CUDA_CHECK_ERROR(cudaSetDevice(1));
    CUDA_CHECK_ERROR(cudaMemcpy(d_D, h_D, size, cudaMemcpyHostToDevice));

    // Create cuBLAS handles
    cublasHandle_t handle0, handle1;
    CUBLAS_CHECK_ERROR(cublasCreate(&handle0));
    CUBLAS_CHECK_ERROR(cublasCreate(&handle1));

    // Create CUDA streams
    cudaStream_t stream0, stream1;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream0));
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream1));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    auto total_on_gpu0 = std::chrono::nanoseconds::zero();

    const int throwaway = 5;  // throw away the first 5 iters

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK_ERROR(cudaSetDevice(0));

        auto cur_start = std::chrono::high_resolution_clock::now();

        if (overlap) {
            // Start async copy from GPU0 to GPU1
            CUDA_CHECK_ERROR(cudaMemcpyPeerAsync(d_E, 1, d_C, 0, size, stream0));

            // Start next iteration's matrix multiplication while copy is ongoing
            CUBLAS_CHECK_ERROR(cublasSgemm(handle0, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));
        } else {
            // Wait for the copy to complete before starting the next iteration
            CUDA_CHECK_ERROR(cudaMemcpyPeer(d_E, 1, d_C, 0, size));

            // Start next iteration's matrix multiplication after copy completes
            CUBLAS_CHECK_ERROR(cublasSgemm(handle0, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));
        }

        auto cur_end = std::chrono::high_resolution_clock::now();
        double cur_dur = std::chrono::duration<double>(cur_end - cur_start).count();
        std::cout << "Iter #" << (i + 1) << " time (" << (overlap ? "Overlapping" : "Synchronous") << "): " << cur_dur << " seconds\n";

        if (i >= throwaway)
            total_on_gpu0 += cur_end - cur_start;

        CUDA_CHECK_ERROR(cudaSetDevice(1));
        CUDA_CHECK_ERROR(cudaStreamSynchronize(stream1));  // Ensure the data is received
    }

    std::cout << "Total time on GPU 0 (" << (overlap ? "Overlapping" : "Synchronous") << "): "
            << std::chrono::duration<double>(total_on_gpu0).count() << " seconds\n";
    std::cout << "Average time on GPU 0 (" << (overlap ? "Overlapping" : "Synchronous") << "): "
            << std::chrono::duration<double>(total_on_gpu0 / (iterations - throwaway)).count() << " seconds\n";

    // Verification
    CUDA_CHECK_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    const auto epsilon = 1e-6;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            if (std::abs(h_C[i * N + j])> epsilon) {
                std::cerr << "Error: C: " << i << ", " << j << " (" << h_C[i * N + j] << ")!" << std::endl;
            }
        }
        if (std::abs(h_C[i * N + i] - ((float) i / N) * ((float) i / N)) > epsilon) {
            std::cerr << "Error: C: " << i << ", " << i << " (" << h_C[i * N + i] << ")!" << std::endl;
        }
        for (int j = i + 1; j < N; ++j) {
            if (std::abs(h_C[i * N + j]) > 1e-6) {
                std::cerr << "Error: C: " << i << ", " << j << " (" << h_C[i * N + j] << ")!" << std::endl;
            }
        }
    }

    // Clean up
    CUDA_CHECK_ERROR(cudaSetDevice(0));
    CUDA_CHECK_ERROR(cudaFree(d_A));
    CUDA_CHECK_ERROR(cudaFree(d_B));
    CUDA_CHECK_ERROR(cudaFree(d_C));

    CUDA_CHECK_ERROR(cudaSetDevice(1));
    CUDA_CHECK_ERROR(cudaFree(d_D));
    CUDA_CHECK_ERROR(cudaFree(d_E));

    CUDA_CHECK_ERROR(cudaStreamDestroy(stream0));
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream1));

    free(h_A);
    free(h_B);
    free(h_D);

    return total_on_gpu0;
}

int main() {
    // const int experiments = 5;
    // for N in [417, 1499], CUBLAS_STATUS_EXECUTION_FAILED error???
    // const int Ns[experiments] = {1500, 2000, 3000, 4000, 5000};
    const int iterations = 100;
    const int N = 12000;

    std::cout << "=============================="
            << std::endl << "N = " << N << std::endl;
    std::cout << "Running synchronous version...\n";
    const auto sync = runExperiment(false, N, iterations);

    std::cout << "Running overlapping version...\n";
    const auto overlap = runExperiment(true, N, iterations);

    const auto speedup = 1 - std::chrono::duration<double>(overlap) / std::chrono::duration<double>(sync);
    // comm: x, compute: y
    // sync: x + y
    // overlap: max(x, y)
    // max(x, y) / x + y >= 0.5
    std::cout << "speedup: " << speedup << std::endl;

    return 0;
}

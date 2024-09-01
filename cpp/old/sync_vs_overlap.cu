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
    std::cout << size << std::endl;
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
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_D = (float*)malloc(size);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_D[i] = static_cast<float>(rand()) / RAND_MAX;
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

    auto start_time = std::chrono::high_resolution_clock::now();

    auto total_on_gpu0 = std::chrono::nanoseconds::zero();

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK_ERROR(cudaSetDevice(0));

        auto cur_start = std::chrono::high_resolution_clock::now();

        // Matrix multiplication on GPU0
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        CUBLAS_CHECK_ERROR(cublasSgemm(handle0, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));

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
        // std::cout << "Iteration #" << (i + 1) << " elapse: " << cur_dur << std::endl;

        total_on_gpu0 += cur_end - cur_start;

        // GPU1: Wait for data, then compute E = C * D
        CUDA_CHECK_ERROR(cudaSetDevice(1));
        CUDA_CHECK_ERROR(cudaStreamSynchronize(stream1));  // Ensure the data is received
        CUBLAS_CHECK_ERROR(cublasSgemm(handle1, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_E, N, d_D, N, &beta, d_E, N));
    }

    CUDA_CHECK_ERROR(cudaSetDevice(0));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());  // Wait for all operations to complete
    CUDA_CHECK_ERROR(cudaSetDevice(1));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());  // Wait for all operations to complete

    auto end_time = std::chrono::high_resolution_clock::now();
    const auto elapse = end_time - start_time;
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total time (" << (overlap ? "Overlapping" : "Synchronous") << "): " << duration << " seconds\n";

    std::cout << "Total time on GPU 0 (" << (overlap ? "Overlapping" : "Synchronous") << "): "
            << std::chrono::duration<double>(total_on_gpu0).count() << " seconds\n";
    std::cout << "Average time on GPU 0 (" << (overlap ? "Overlapping" : "Synchronous") << "): "
            << std::chrono::duration<double>(total_on_gpu0 / iterations).count() << " seconds\n";

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

    return elapse;
}

int main() {
    const int experiments = 5;
    // for N in [417, 1499], CUBLAS_STATUS_EXECUTION_FAILED error???
    const int Ns[experiments] = {1000, 2000, 3000, 4000, 5000};
    const int iterations = 100;

    for (int i = 0; i < experiments; i++) {
        const int N = Ns[i];
        std::cout << "=============================="
                << std::endl << "N = " << N << std::endl;
        std::cout << "Running synchronous version...\n";
        const auto sync = runExperiment(false, N, iterations);

        std::cout << "Running overlapping version...\n";
        const auto overlap = runExperiment(true, N, iterations);

        const auto speedup = 1 - std::chrono::duration<double>(overlap) / std::chrono::duration<double>(sync);
        std::cout << "speedup: " << speedup << std::endl;
    }

    return 0;
}

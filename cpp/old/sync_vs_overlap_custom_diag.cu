#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

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

std::chrono::nanoseconds runExperiment(bool overlap, int N, int iterations) {
    size_t size = N * N * sizeof(float);

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
    float *h_C = (float*)malloc(size);
    float *h_D = (float*)malloc(size);
    float *h_E = (float*)malloc(size);
    for (int i = 0; i < N; ++i) {
        h_A[i * N + i] = (float) i / N;
        // std::cout << (float) i / N;
        h_B[i * N + i] = (float) i / N;
        h_D[i * N + i] = (float) i / N;
    }

    // Copy matrices A, B, D from host to device
    CUDA_CHECK_ERROR(cudaSetDevice(0));
    CUDA_CHECK_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    CUDA_CHECK_ERROR(cudaSetDevice(1));
    CUDA_CHECK_ERROR(cudaMemcpy(d_D, h_D, size, cudaMemcpyHostToDevice));

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
        matmul<<<gridDim, blockDim, 0, stream0>>>(d_A, d_B, d_C, N);

        if (overlap) {
            // Start async copy from GPU0 to GPU1
            CUDA_CHECK_ERROR(cudaMemcpyPeerAsync(d_E, 1, d_C, 0, size, stream0));

            // Start next iteration's matrix multiplication while copy is ongoing
            matmul<<<gridDim, blockDim, 0, stream0>>>(d_A, d_B, d_C, N);
        } else {
            // Wait for the copy to complete before starting the next iteration
            CUDA_CHECK_ERROR(cudaMemcpyPeer(d_E, 1, d_C, 0, size));

            // Start next iteration's matrix multiplication after copy completes
            matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        }

        auto cur_end = std::chrono::high_resolution_clock::now();
        double cur_dur = std::chrono::duration<double>(cur_end - cur_start).count();
        // std::cout << "Iteration #" << (i + 1) << " elapse: " << cur_dur << std::endl;

        total_on_gpu0 += cur_end - cur_start;

        // GPU1: Wait for data, then compute E = C * D
        CUDA_CHECK_ERROR(cudaSetDevice(1));
        CUDA_CHECK_ERROR(cudaStreamSynchronize(stream1));  // Ensure the data is received
        matmul<<<gridDim, blockDim, 0, stream1>>>(d_E, d_D, d_E, N);
    }

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());  // Wait for all operations to complete

    auto end_time = std::chrono::high_resolution_clock::now();
    const auto elapse = end_time - start_time;
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total time (" << (overlap ? "Overlapping" : "Synchronous") << "): " << duration << " seconds\n";

    std::cout << "Total time on GPU 0 (" << (overlap ? "Overlapping" : "Synchronous") << "): "
            << std::chrono::duration<double>(total_on_gpu0).count() << " seconds\n";
    std::cout << "Average time on GPU 0 (" << (overlap ? "Overlapping" : "Synchronous") << "): "
            << std::chrono::duration<double>(total_on_gpu0 / iterations).count() << " seconds\n";

    // Verification
    CUDA_CHECK_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_E, d_E, size, cudaMemcpyDeviceToHost));
    const auto epsilon = 1e-6;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            if (h_C[i * N + j] != 0) {
                std::cerr << "Error: C: " << i << ", " << j << "!" << std::endl;
            }
        }
        if (h_C[i * N + i] - ((float) i / N) * ((float) i / N) > epsilon) {
            std::cerr << "Error: C: " << i << ", " << i << "!" << std::endl;
        }
        for (int j = i + 1; j < N; ++j) {
            if (h_C[i * N + j] != 0) {
                std::cerr << "Error: C: " << i << ", " << j << "!" << std::endl;
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            if (h_E[i * N + j] != 0) {
                std::cerr << "Error: E: " << i << ", " << j << "!" << std::endl;
            }
        }
        if (h_E[i * N + i] - ((float) i / N) * ((float) i / N) * ((float) i / N) > epsilon) {
            std::cerr << "Error: E: " << i << ", " << i << "!" << std::endl;
        }
        for (int j = i + 1; j < N; ++j) {
            if (h_E[i * N + j] != 0) {
                std::cerr << "Error: E: " << i << ", " << j << "!" << std::endl;
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

    return elapse;
}

int main() {
    const int experiments = 5;
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

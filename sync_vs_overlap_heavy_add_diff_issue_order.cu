#include <iostream>
#include <chrono>

inline void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void add(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        for (int i = 0; i < 32; i++) {
            float sinA = sinf(A[tid]);
            float cosA = cosf(A[tid]);
            float idA = sqrtf(sinA * sinA + cosA * cosA) * A[tid];
            float sinB = sinf(B[tid]);
            float cosB = cosf(B[tid]);
            float idB = sqrtf(sinB * sinB + cosB * cosB) * B[tid];
            float added = idA + idB;
            float sinAdded = sinf(added);
            float cosAdded = cosf(added);
            float idAdded = sqrtf(sinAdded * sinAdded + cosAdded * cosAdded) * added;
            C[tid] = idAdded;
        }
    }
}

void maxError(float *output, int iterations, int N, bool isC) {
    float maxErr = 0;
    int maxI = -1;
    const float epsilon = 1e-6;
    // basic comp:
    float expected = isC ? 3.0f : 6.0f;
    // heavy comp:
    // float expected = 4 * iterations + 3;
    // heavier comp:
    // float expected = 8 * iterations + 3;
    // if (!isC) expected += 3;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(output[i] - expected);
        if (diff > maxErr) {
            maxErr = diff;
            maxI = i;
        }
        if (diff > epsilon) {
            std::cout << "Error: " << (isC ? "C" : "F") << ": " << i << " (" << output[i] << ")!" << std::endl;
        }
    }
    std::cout << "Max error: " << maxErr << std::endl;
    if (maxI != -1) {
        std::cout << "at " << maxI << ": " << output[maxI] << std::endl;
    }
}

float experiment(bool overlap, int N, int iterations) {
    // if (iterations % 2 != 0) {
    //     std::cout << "iterations must be even" << std::endl;
    //     iterations--;
    // }
    size_t size = N * sizeof(float);
    int block = 256;
    int grid = (N + block - 1) / block;

    float *d_A, *d_B, *d_C, *d_X;
    checkCudaError(cudaSetDevice(0));
    checkCudaError(cudaMalloc((void**)&d_A, size));
    checkCudaError(cudaMalloc((void**)&d_B, size));
    checkCudaError(cudaMalloc((void**)&d_C, size));
    checkCudaError(cudaMalloc((void**)&d_X, size));

    float *d_D, *d_E, *d_F, *d_Y;
    checkCudaError(cudaSetDevice(1));
    checkCudaError(cudaMalloc((void**)&d_D, size));
    checkCudaError(cudaMalloc((void**)&d_E, size));
    checkCudaError(cudaMalloc((void**)&d_F, size));
    checkCudaError(cudaMalloc((void**)&d_Y, size));

    float *h_A, *h_B, *h_C, *h_D, *h_F;
    checkCudaError(cudaMallocHost((void**)&h_A, size));
    checkCudaError(cudaMallocHost((void**)&h_B, size));
    checkCudaError(cudaMallocHost((void**)&h_C, size));
    checkCudaError(cudaMallocHost((void**)&h_D, size));
    checkCudaError(cudaMallocHost((void**)&h_F, size));

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
        h_C[i] = -100.0f;
        h_D[i] = 3.0f;
    }

    cudaEvent_t startEvent, endEvent,
            copyCEvent, copyXEvent,
            compCEvent, compXEvent,
            compFEvent, compYEvent;
    checkCudaError(cudaEventCreate(&startEvent));
    checkCudaError(cudaEventCreate(&endEvent));

    cudaStream_t compute0, compute1, copyStream;
    checkCudaError(cudaSetDevice(0));
    checkCudaError(cudaStreamCreate(&compute0));
    checkCudaError(cudaEventCreate(&compCEvent));
    checkCudaError(cudaEventCreate(&compXEvent));
    checkCudaError(cudaStreamCreate(&copyStream));
    checkCudaError(cudaEventCreate(&copyCEvent));
    checkCudaError(cudaEventCreate(&copyXEvent));
    checkCudaError(cudaSetDevice(1));
    checkCudaError(cudaStreamCreate(&compute1));
    checkCudaError(cudaEventCreate(&compFEvent));
    checkCudaError(cudaEventCreate(&compYEvent));

    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

    checkCudaError(cudaSetDevice(1));
    checkCudaError(cudaMemcpy(d_D, h_D, size, cudaMemcpyHostToDevice));

    checkCudaError(cudaEventRecord(startEvent, 0));
    const auto chronoStart = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        checkCudaError(cudaSetDevice(0));

        checkCudaError(cudaStreamWaitEvent(compute0, copyCEvent, 0));
        add<<<grid, block, 0, compute0>>>(d_A, d_B, d_C, N);  // C_1 = A_0 + B_0
        checkCudaError(cudaGetLastError());

        if (overlap) {
            checkCudaError(cudaEventRecord(compCEvent, compute0));
        } else {
            checkCudaError(cudaMemcpyPeerAsync(d_E, 1, d_C, 0, size, compute0));
            checkCudaError(cudaEventRecord(copyCEvent, compute0));
        }

        checkCudaError(cudaStreamWaitEvent(compute0, copyXEvent, 0));
        add<<<grid, block, 0, compute0>>>(d_A, d_B, d_X, N);  // X_1 = A_0 + B_0
        checkCudaError(cudaGetLastError());

        if (overlap) {
            checkCudaError(cudaEventRecord(compXEvent, compute0));
        } else {
            checkCudaError(cudaMemcpyPeerAsync(d_E, 1, d_X, 0, size, compute0));
            checkCudaError(cudaEventRecord(copyXEvent, compute0));
        }

        if (overlap) {
            checkCudaError(cudaStreamWaitEvent(copyStream, compCEvent));
            checkCudaError(cudaMemcpyPeerAsync(d_E, 1, d_C, 0, size, copyStream));
            checkCudaError(cudaEventRecord(copyCEvent, copyStream));

            checkCudaError(cudaStreamWaitEvent(copyStream, compXEvent));
            checkCudaError(cudaMemcpyPeerAsync(d_E, 1, d_X, 0, size, copyStream));
            checkCudaError(cudaEventRecord(copyXEvent, copyStream));
        }
        checkCudaError(cudaSetDevice(1));

        checkCudaError(cudaStreamWaitEvent(compute1, copyCEvent, 0));
        checkCudaError(cudaStreamWaitEvent(compute1, compFEvent, 0));
        add<<<grid, block, 0, compute1>>>(d_D, d_E, d_F, N);
        checkCudaError(cudaGetLastError());
        checkCudaError(cudaEventRecord(compFEvent, compute1));

        checkCudaError(cudaStreamWaitEvent(compute1, copyXEvent, 0));
        checkCudaError(cudaStreamWaitEvent(compute1, compYEvent, 0));
        add<<<grid, block, 0, compute1>>>(d_D, d_E, d_Y, N);
        checkCudaError(cudaGetLastError());
        checkCudaError(cudaEventRecord(compYEvent, compute1));
    }

    checkCudaError(cudaEventRecord(endEvent, 0));
    checkCudaError(cudaEventSynchronize(endEvent));

    const auto chronoEnd = std::chrono::high_resolution_clock::now();
    const float chronoElapse = std::chrono::duration<float>(chronoEnd - chronoStart).count();
    std::cout << "chrono time: " << chronoElapse << std::endl;

    float elapse;
    checkCudaError(cudaEventElapsedTime(&elapse, startEvent, endEvent));

    std::cout << "Total time (" << (overlap ? "Overlapping" : "Synchronous") << "): " << elapse << " ms\n";

    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(h_F, d_F, size, cudaMemcpyDeviceToHost));
    // checkCudaError(cudaStreamSynchronize(compute0));
    // checkCudaError(cudaStreamSynchronize(compute1));

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
    checkCudaError(cudaEventDestroy(compCEvent));
    checkCudaError(cudaEventDestroy(compXEvent));
    checkCudaError(cudaEventDestroy(compFEvent));
    checkCudaError(cudaEventDestroy(compYEvent));
    checkCudaError(cudaEventDestroy(copyCEvent));
    checkCudaError(cudaEventDestroy(copyXEvent));

    checkCudaError(cudaStreamDestroy(compute0));
    checkCudaError(cudaStreamDestroy(compute1));

    checkCudaError(cudaFreeHost(h_A));
    checkCudaError(cudaFreeHost(h_B));
    checkCudaError(cudaFreeHost(h_C));
    checkCudaError(cudaFreeHost(h_D));
    checkCudaError(cudaFreeHost(h_F));

    return elapse;
}

int main() {
    // const int experiments = 8;
    // const int iterations = 100;
    // int base = 1;

    const int experiments = 1;
    const int iterations = 10;
    int base = 100000;

    for (int i = 0; i < experiments; i++) {
        const int N = base;
        base *= 10;
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

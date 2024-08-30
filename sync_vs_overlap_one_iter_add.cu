#include <iostream>

inline void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void add(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        C[tid] = A[tid] + B[tid];
}

void maxError(float *output, int iterations, int N, bool isC) {
    float maxErr = 0;
    int maxI = -1;
    const float epsilon = 1e-6;
    for (int i = 0; i < N; ++i) {
        float expected = isC ? 3.0f : 6.0f;
        float diff = std::abs(output[i] - expected);
        if (diff > maxErr) {
            maxErr = diff;
            maxI = i;
        }
        if (diff > epsilon) {
            std::cerr << "Error: " << (isC ? "C" : "F") << ": " << i << " (" << output[i] << ")!" << std::endl;
        }
    }
    std::cout << "Max error: " << maxErr << std::endl;
    if (maxI != -1) {
        std::cout << "at " << maxI << ": " << output[maxI] << std::endl;
    }
}

float experiment(bool overlap, int N, int iterations) {
    size_t size = N * sizeof(float);
    int block = 256;
    int grid = (N + block - 1) / block;

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

    cudaEvent_t startEvent, endEvent, copyEvent;
    checkCudaError(cudaEventCreate(&startEvent));
    checkCudaError(cudaEventCreate(&endEvent));

    cudaStream_t stream0, stream1;
    checkCudaError(cudaSetDevice(0));
    checkCudaError(cudaStreamCreate(&stream0));
    checkCudaError(cudaEventCreate(&copyEvent));
    checkCudaError(cudaSetDevice(1));
    checkCudaError(cudaStreamCreate(&stream1));

    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

    checkCudaError(cudaSetDevice(1));
    checkCudaError(cudaMemcpy(d_D, h_D, size, cudaMemcpyHostToDevice));

    checkCudaError(cudaEventRecord(startEvent, 0));

    for (int i = 0; i < iterations; ++i) {
        checkCudaError(cudaSetDevice(0));
        // puts("first add:");
        add<<<grid, block, 0, stream0>>>(d_A, d_B, d_C, N);
        checkCudaError(cudaGetLastError());
        // puts("first add finished");
        if (overlap) {
            // checkCudaError(cudaMemcpyPeerAsync(d_E, 1, d_C, 0, size, stream0));
            // puts("overlap copy:");
            checkCudaError(cudaMemcpyAsync(d_E, d_C, size, cudaMemcpyDeviceToDevice, stream0));
            // puts("overlap copy launched");
            checkCudaError(cudaEventRecord(copyEvent, stream0));
        } else {
            // puts("sync copy:");
            // checkCudaError(cudaStreamSynchronize(stream0));
            // checkCudaError(cudaMemcpyPeer(d_E, 1, d_C, 0, size));
            // checkCudaError(cudaMemcpy(d_E, d_C, size, cudaMemcpyDeviceToDevice));
            checkCudaError(cudaMemcpyAsync(d_E, d_C, size, cudaMemcpyDeviceToDevice, stream0));
            // puts("sync copy launched");
            // puts("syncing stream0:");
            checkCudaError(cudaStreamSynchronize(stream0));
            // puts("synced");
            // checkCudaError(cudaEventRecord(copyEvent, stream0));
            // checkCudaError(cudaStreamSynchronize(0));
            // puts("recording copy event:");
            checkCudaError(cudaEventRecord(copyEvent, stream0));
            // puts("recorded");
        }
        checkCudaError(cudaSetDevice(1));
        checkCudaError(cudaStreamWaitEvent(stream1, copyEvent, 0));
        // checkCudaError(cudaEventSynchronize(copyEvent));

        // puts("second add:");
        add<<<grid, block, 0, stream1>>>(d_D, d_E, d_F, N);
        checkCudaError(cudaGetLastError());
    }

    checkCudaError(cudaStreamSynchronize(stream0));
    checkCudaError(cudaStreamSynchronize(stream1));
    checkCudaError(cudaEventRecord(endEvent, 0));
    checkCudaError(cudaEventSynchronize(endEvent));
    // checkCudaError(cudaSetDevice(0));
    // checkCudaError(cudaDeviceSynchronize());
    // checkCudaError(cudaSetDevice(1));
    // checkCudaError(cudaDeviceSynchronize());

    float elapse;
    checkCudaError(cudaEventElapsedTime(&elapse, startEvent, endEvent));

    std::cout << "Total time (" << (overlap ? "Overlapping" : "Synchronous") << "): " << elapse << " ms\n";

    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(h_F, d_F, size, cudaMemcpyDeviceToHost));
    // checkCudaError(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream0));
    // checkCudaError(cudaMemcpyAsync(h_F, d_F, size, cudaMemcpyDeviceToHost, stream1));
    checkCudaError(cudaStreamSynchronize(stream0));
    checkCudaError(cudaStreamSynchronize(stream1));

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
    checkCudaError(cudaEventDestroy(copyEvent));

    checkCudaError(cudaStreamDestroy(stream0));
    checkCudaError(cudaStreamDestroy(stream1));

    checkCudaError(cudaFreeHost(h_A));
    checkCudaError(cudaFreeHost(h_B));
    checkCudaError(cudaFreeHost(h_C));
    checkCudaError(cudaFreeHost(h_D));
    checkCudaError(cudaFreeHost(h_F));

    return elapse;
}

int main() {
    const int experiments = 1;
    const int Ns[experiments] = {10};
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

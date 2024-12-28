#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>

__global__ void generateXORWOW(float *d_data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandStateXORWOW_t state;
    curand_init(1234, tid, 0, &state);

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float randomValue = curand_uniform(&state);

        d_data[i] = 2.0f * randomValue - 1.0f;
    }
}

__global__ void generateMRG32k3a(float *d_data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandStateMRG32k3a_t state;
    curand_init(2134, tid, 0, &state);

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float randomValue = curand_uniform(&state);

        d_data[i] = 2.0f * randomValue - 1.0f;
    }
}

__global__ void generatePhilox_4x32_10(float *d_data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(3124, tid, 0, &state);

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float randomValue = curand_uniform(&state);

        d_data[i] = 2.0f * randomValue - 1.0f;
    }
}

__global__ void generateSobol(float *d_data, curandDirectionVectors32_t *directionVectors, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandStateSobol32_t state;

    curand_init(*directionVectors, tid, &state);

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float randomValue = curand_uniform(&state);

        d_data[i] = 2.0f * randomValue - 1.0f;
    }
}

void calculate_pi(float *d_data_1, float *d_data_2, int n)
{
    float counter = 0.0f;

    float pi = 0.0f;

    for (int i = 0; i < n; i++)
    {
        // std::cout << "Counter first pair: " << d_data_1[i] << "; " << d_data_2[i] << std::endl;
        if(((d_data_1[i] * d_data_1[i]) + (d_data_2[i] * d_data_2[i])) <= 1.0f)
        {
            counter++;
        }
    }

    std::cout << "Counter value: " << counter << std::endl;

    pi = 4.0f * (counter / (float)n);

    std::cout << "PI value: " << pi << std::endl;
}

int main(void)
{
    const int n = 1000000;
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    float *h_data_1 = new float[n];
    float *d_data_1;

    float *h_data_2 = new float[n];
    float *d_data_2;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // XORWOW

    cudaMalloc((void**)&d_data_1, n * sizeof(float));
    cudaMalloc((void**)&d_data_2, n * sizeof(float));

    cudaEventRecord(start);
    generateXORWOW<<<numBlocks, blockSize>>>(d_data_1, n);

    generateXORWOW<<<numBlocks, blockSize>>>(d_data_2, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution time XORWOW: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_data_1, d_data_1, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data_2, d_data_2, n * sizeof(float), cudaMemcpyDeviceToHost);

    calculate_pi(h_data_1, h_data_2, n);

    delete[] h_data_1;
    delete[] h_data_2;
    cudaFree(d_data_1);
    cudaFree(d_data_2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // MRG32k3a

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_data_1 = new float[n];
    h_data_2 = new float[n];

    cudaMalloc((void**)&d_data_1, n * sizeof(float));
    cudaMalloc((void**)&d_data_2, n * sizeof(float));

    cudaEventRecord(start);
    generateMRG32k3a<<<numBlocks, blockSize>>>(d_data_1, n);

    generateMRG32k3a<<<numBlocks, blockSize>>>(d_data_2, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution time MRG32k3a: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_data_1, d_data_1, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data_2, d_data_2, n * sizeof(float), cudaMemcpyDeviceToHost);

    calculate_pi(h_data_1, h_data_2, n);

    delete[] h_data_1;
    delete[] h_data_2;
    cudaFree(d_data_1);
    cudaFree(d_data_2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Philox_4x32_10

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_data_1 = new float[n];
    h_data_2 = new float[n];

    cudaMalloc((void**)&d_data_1, n * sizeof(float));
    cudaMalloc((void**)&d_data_2, n * sizeof(float));

    cudaEventRecord(start);
    generatePhilox_4x32_10<<<numBlocks, blockSize>>>(d_data_1, n);

    generatePhilox_4x32_10<<<numBlocks, blockSize>>>(d_data_2, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution time Philox_4x32_10: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_data_1, d_data_1, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data_2, d_data_2, n * sizeof(float), cudaMemcpyDeviceToHost);

    calculate_pi(h_data_1, h_data_2, n);

    delete[] h_data_1;
    delete[] h_data_2;
    cudaFree(d_data_1);
    cudaFree(d_data_2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Sobol

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    curandDirectionVectors32_t *directionVectors;

    const size_t numDirectionVectors = 32;

    cudaMallocHost((void**)&directionVectors, numDirectionVectors * 20000);

    curandGetDirectionVectors32(&directionVectors, CURAND_DIRECTION_VECTORS_32_JOEKUO6);

    h_data_1 = new float[n];
    h_data_2 = new float[n];

    cudaMalloc((void**)&d_data_1, n * sizeof(float));
    cudaMalloc((void**)&d_data_2, n * sizeof(float));

    cudaEventRecord(start);
    generateSobol<<<numBlocks, blockSize>>>(d_data_1, directionVectors, n);

    generateSobol<<<numBlocks, blockSize>>>(d_data_2, directionVectors, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution time Sobol: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_data_1, d_data_1, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data_2, d_data_2, n * sizeof(float), cudaMemcpyDeviceToHost);

    calculate_pi(h_data_1, h_data_2, n);

    delete[] h_data_1;
    delete[] h_data_2;
    cudaFree(d_data_1);
    cudaFree(d_data_2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(directionVectors);

    return 0;
}

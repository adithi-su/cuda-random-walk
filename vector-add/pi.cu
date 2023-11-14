#include <iostream>
#include <ctime>
#include <curand_kernel.h>

#define NUM_BLOCKS 256
#define THREADS_PER_BLOCK 256
#define NUM_POINTS (NUM_BLOCKS * THREADS_PER_BLOCK)
#define RADIUS 1.0

__global__ void calculatePi(float* results, unsigned int seed) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, tid, 0, &state);

    int pointsInsideCircle = 0;

    for (int i = 0; i < NUM_POINTS; ++i) {
        float x = curand_uniform(&state) * RADIUS;
        float y = curand_uniform(&state) * RADIUS;

        if (x * x + y * y <= RADIUS * RADIUS) {
            pointsInsideCircle++;
        }
    }

    results[tid] = 4.0 * pointsInsideCircle / NUM_POINTS;
}

int main() 
{
    float* d_results;
    float* h_results = new float[NUM_BLOCKS * THREADS_PER_BLOCK];

    cudaMalloc((void**)&d_results, sizeof(float) * NUM_BLOCKS * THREADS_PER_BLOCK);

    calculatePi<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_results, time(NULL));
    cudaMemcpy(h_results, d_results, sizeof(float) * NUM_BLOCKS * THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);

    float pi = 0;
    for (int i = 0; i < NUM_BLOCKS * THREADS_PER_BLOCK; ++i) {
        pi += h_results[i];
    }
    pi /= NUM_BLOCKS * THREADS_PER_BLOCK;

    std::cout << "Estimated value of PI: " << pi << std::endl;

    cudaFree(d_results);
    delete[] h_results;

    return 0;
}


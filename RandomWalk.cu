/*
Author: Adithi Upadhya
Class: ECE6122
Last Date Modified: 11/08/2023
Description: CUDA-based 2D Random Walk Simulation
*/

#include <iostream>
#include <vector>
#include <ctime>
#include <curand_kernel.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//#define NUM_BLOCKS 256
//#define THREADS_PER_BLOCK 256

__global__ void RandomWalk(uint64_t* position_x, uint64_t* position_y, uint64_t num_walkers, uint64_t num_steps, unsigned int seed)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, tid, 0, &state);
    if (tid < num_walkers)
    {
        int x = 0; 
        int y = 0;
        for (unsigned int i = 0; i < num_steps; ++i)
        {
            float randv = curand_uniform(&state); 
            if (randv < 0.25)
                x -= 1; //go left
            else if (randv >= 0.25 && randv < 0.5)
                x += 1; //go right
            else if (randv >= 0.5 && randv < 0.75)
                y += 1; //go up
            else
                y -= 1; //go down
        }
        position_x[tid] = x; 
        position_y[tid] = y; 
    }
}



int main(int argc, char* argv[])
{   
    uint64_t num_walkers, num_steps; 
    cudaEvent_t startEvent, stopEvent; 
    float elapsed_time;

    //default values
    num_walkers = 10000;
    num_steps = 1000000;
    
    //flags to track options, argc contains no. of parameters
    for (int i = 1; i < argc; i += 2) 
    {
        if (argv[i][1] == 'W') 
        {
            if (!isdigit(argv[i + 1][0]))
            {
                std::cerr << "Invalid input" << std::endl;
                return 1;
            }
            else
                num_walkers = atoi(argv[i + 1]);
        }
        else if (argv[i][1] == 'I') {
            if (!isdigit(argv[i + 1][0]))
            {
                std::cerr << "Invalid input" << std::endl;
                return 1; 
            }
            else
                num_steps = atoi(argv[i + 1]);
        }
        else 
        {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            return 1;
        }
    }

    //unsigned int seed = static_cast<unsigned int>(time(NULL));
    float avg_dist1 = 0.0;
    float avg_dist2 = 0.0;
    float avg_dist3 = 0.0;

    std::cout << "Number of walkers: " << num_walkers << "\n";
    std::cout << "Number of steps: " << num_steps << "\n"; 

    //kernel dimensions
    int block_size = 256;
    int grid_size = ((num_walkers + block_size) / block_size);

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    //-----------------------------------------------function 1-------------------------------------------------------------------------------------
    uint64_t* pageable_pos_x; //host memory
    uint64_t* pageable_pos_y; //host memory
    uint64_t* d_x; //device memory
    uint64_t* d_y; //device memory

    cudaEventRecord(startEvent, 0);

    pageable_pos_x = (uint64_t*)malloc(sizeof(uint64_t) * num_walkers); //Allocate host memory, pageable
    pageable_pos_y = (uint64_t*)malloc(sizeof(uint64_t) * num_walkers); //Allocate host memory, pageable
    
    memset(pageable_pos_x, 0, sizeof(uint64_t) * num_walkers);
    memset(pageable_pos_y, 0, sizeof(uint64_t) * num_walkers);
    
    cudaMalloc((uint64_t**)&d_x, sizeof(uint64_t) * num_walkers); //Allocate device memory
    cudaMalloc((uint64_t**)&d_y, sizeof(uint64_t) * num_walkers);

    RandomWalk <<<grid_size, block_size >>> (d_x, d_y, num_walkers, num_steps, time(NULL)); //Execute kernel

    cudaMemcpy(pageable_pos_x, d_x, sizeof(uint64_t) * num_walkers, cudaMemcpyDeviceToHost); //Transfer data back to host memory
    cudaMemcpy(pageable_pos_y, d_y, sizeof(uint64_t) * num_walkers, cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < num_walkers; ++i)
    {
        avg_dist1 += sqrt(pageable_pos_x[i] * pageable_pos_x[i] + pageable_pos_y[i] * pageable_pos_y[i]);
    }
    avg_dist1 /= num_walkers; 
    
    cudaFree(pageable_pos_x);
    cudaFree(pageable_pos_y);
    cudaFree(d_x);
    cudaFree(d_y);
    
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent); 
    cudaEventElapsedTime(&elapsed_time, startEvent, stopEvent);

    std::cout << "Normal CUDA memory allocation:\n";
    std::cout << "    Time to calculate(microsec): " << elapsed_time*1000.0 << "\n";
    std::cout << "    Average distance from origin: " << avg_dist1 << "\n";
   
    //-----------------------------------------------function 2------------------------------------------------------------------------------------- 
    
    uint64_t* pinned_pos_x;
    uint64_t* pinned_pos_y;
    uint64_t* d2_x; //device memory
    uint64_t* d2_y; //device memory
    
    cudaEventRecord(startEvent, 0);

    cudaMalloc((uint64_t**)&d2_x, sizeof(uint64_t) * num_walkers); //Allocate device memory
    cudaMalloc((uint64_t**)&d2_y, sizeof(uint64_t) * num_walkers);

    cudaMallocHost((void**)&pinned_pos_x, sizeof(uint64_t) * num_walkers); //host, pinned
    cudaMallocHost((void**)&pinned_pos_y, sizeof(uint64_t) * num_walkers);

    memset(pinned_pos_x, 0, sizeof(uint64_t) * num_walkers);
    memset(pinned_pos_y, 0, sizeof(uint64_t) * num_walkers);

    RandomWalk << <grid_size, block_size >> > (d2_x, d2_y, num_walkers, num_steps, time(NULL)); //Execute kernel

    cudaMemcpy(pinned_pos_x, d2_x, sizeof(uint64_t) * num_walkers, cudaMemcpyDeviceToHost); //Transfer data back to host memory
    cudaMemcpy(pinned_pos_y, d2_y, sizeof(uint64_t) * num_walkers, cudaMemcpyDeviceToHost);
    
    for (unsigned int i = 0; i < num_walkers; ++i)
        avg_dist2 += sqrt(pinned_pos_x[i] * pinned_pos_x[i] + pinned_pos_y[i]  * pinned_pos_y[i]);
    avg_dist2 /= num_walkers;

    cudaFreeHost(pinned_pos_x);
    cudaFreeHost(pinned_pos_y);
    cudaFree(d2_x);
    cudaFree(d2_y);
   
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsed_time, startEvent, stopEvent);

    std::cout << "Pinned CUDA memory Allocation:\n";
    std::cout << "    Time to calculate(microsec): " << elapsed_time * 1000.0 << "\n";
    std::cout << "    Average distance from origin: " << avg_dist2 << "\n";
    
    //-----------------------------------------------function 3-------------------------------------------------------------------------------------
    uint64_t* m_positions_x; //managed memory
    uint64_t* m_positions_y;
    
    cudaEventRecord(startEvent, 0);

    cudaMallocManaged((void**)&m_positions_x, sizeof(uint64_t) * num_walkers);
    cudaMallocManaged((void**)&m_positions_y, sizeof(uint64_t) * num_walkers);

    RandomWalk <<<grid_size, block_size >>> (m_positions_x, m_positions_y, num_walkers, num_steps, time(NULL));
    cudaDeviceSynchronize();

    for (unsigned int i = 0; i < num_walkers; ++i)
        avg_dist3 += sqrt(m_positions_x[i] * m_positions_x[i] + m_positions_y[i] * m_positions_y[i]);
    avg_dist3 /= num_walkers;

    cudaFreeHost(m_positions_x);
    cudaFreeHost(m_positions_y);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsed_time, startEvent, stopEvent);

    std::cout << "Managed CUDA memory allocation:\n";
    std::cout << "    Time to calculate(microsec): " << elapsed_time * 1000.0 << "\n";
    std::cout << "    Average distance from origin: " << avg_dist3 << "\n";
    std::cout << "Bye!" << "\n";

    //clean up
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    
    return 0;

}


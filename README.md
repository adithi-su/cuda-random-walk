# CUDA

Problem statement: A CUDA program to simulate a 2D random walk with command line flags to distinguish Number Walkers (-W) and (-I) for number of steps. All the walkers start at the origin (0, 0). The application implements three functions that internally use different memory models to perform the calculations:
1. cudaMalloc
2. cudaMallocHost
3. cudaMallocManaged
   
Outputs to the console screen include the total time each function took to perform the calculations and the average distance of the walkers from the origin.

Instructions to run the code:
```
module load gcc/10.3 cuda/11.7
nvcc *.cu -o RandomWalk
./RandomWalk -W 10000 -I 10000
```
## NOTES
- add __global__ - kernel function that runs on GPU
- &d_a - > d is device mem
- **kernel config <<<M ,T >>>** : kernel launches with a grid of M thread blocks, each thread block has T parallel threads → 1,1 - 1 block with 1 thread
- threads organized into “**thread block**”
- kernel can launch multiple thread blocks, organized into a “grid”
- CUDA GPUs run kernels using blocks of threads that are a **multiple of 32**
- **vector_add <<< 1, 256 >>> (d_out, d_a, d_b, N);**
    - **threadIdx.x** - contains index of thread within the block(range 0 - 255)
    - **blockDim.x** - contains size of thread block (number of threads in the block) (256 here)
    - **blockIdx.x** - contains index of block within grid
    - **gridDim.x** - size of grid
- vector_add is executed on the GPU, need to copy it back to host mem
- loop starts from kth element, iterates through the array with a loop stride of 256.
    - 0th iteration: kth thread computes addition of kth element
    - next iteration: kth thread computes addition of (k+256)th element

- CUDA GPUs have several parallel processors called SM (**streaming multiprocessors**), each consisting of multiple parallel processors and can run multiple concurrent threads
- **cudaGetDeviceCount()** - count of all CUDA devices

**EXAMPLE 2:**

- when you have multiple threads, divide the kernel function
- the same example with 256 threads is still slower than the just CPU - this is because the GPU involves a lot of mem transfer from host to user machine
    - profiling

**nsys** - new NVIDIA profiler, gives all stats

**EXAMPLE 3:**
custom grid size
trying to make use of the GPU such that each thread can do a single element

**NPP**

https://docs.nvidia.com/cuda/npp/index.html

- NVIDIA 2D image and signal performance primtives
- **boxFilterNPP** - computes avg pixel values of the pixels under a rectangular mask

nppiFilterBoxBorder_8u_C1R (const Npp8u *pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)

- **cannyEdgeDetectorNPP** - edge detection operator (uses a multi-stage algorithm to detect wide range of edges in imgs)

https://en.wikipedia.org/wiki/Canny_edge_detector

**CUDA STREAMS**

http://horacio9573.no-ip.org/cuda/group__CUDART__STREAM.html

**Stream**: sequence of operations executed in order on GPU 

- ex: memory transfers, kernels
- CUDA operations are placed within a stream, by default synchronous
- queue of device work
- host places work in queue and continues on immediately
- device schedules work from streams when resources are free
- operations within same stream are ordered (FIFO) and cannot overlap
- operations within diff streams are unordered and can overlap

**Functions:**

- cudaError_T cudaStreamCreate (cudaStream_t *pStream) - create an async stream
- cudaError_T cudaStreamDestroy (cudaStream_t stream) - destroy and clean up async stream
- cudaError_T cudaStreamQuery (cudaStream_t stream) - queries async stream for completion status
- cudaError_T cudaStreamSynchronize (cudaStream_t stream) - waits for stream tasks to complete
- cudaError_T cudaStreamWaitEvent (cudaStream_t stream, cudaEvent_t event, unsigned int flags) - make a compute stream wait on an event

**MEMORY TRANSFERS**

cudaMemcpyXXXX() - is blocking

cudaMemcpyXXXXAsync() - can be non blocking if non zero stream is defined; async wrt host - call may return before copy is complete

**Concurrency requirements:**

- CUDA operations must be in different, non-0, streams
- cudaMemcpyAsync with host from 'pinned' memory (allocated using cudaMallocHost instead of malloc)
    - Page-locked memory
    - Allocated using cudaMallocHost() or cudaHostAlloc()
- Sufficient resources must be available
    - cudaMemcpyAsyncs in different directions
    - Device resources

http://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

**Optimize data transfers in CUDA C/C++:**

guidelines for host-device data transfers:

1. Minimize the amount of data transferred between host and device when possible, even if that means running kernels on the GPU that get little or no speed-up compared to running them on the host CPU.

2. Higher bandwidth is possible between the host and the device when using page-locked (or “pinned”) memory.

3. Batching many small transfers into one larger transfer performs much better because it eliminates most of the per-transfer overhead.

4. Data transfers between the host and device can sometimes be overlapped with kernel execution and other data transfers
---
- for high computation problems, integrated device (on devie GPU - limited capabilities) should be set to False
- by using pinned memory(faster), we’re basically saving some allocations and copy steps

- why not always use pinned mem? - amt of pin mem available is smaller than pageable

- default malloc allocates pageable memory 

- call the function cudaMallocHost separately for pinned:

- cudaMalloc allocates mem on device

- modern GPUs have **managed memory**  - shared between host and device - don’t have to transfer back and forth

- might have to synchronize while using managed 

- memset - give it starting address, value and bytes (works better than using a loop)
- make sure to free up resources

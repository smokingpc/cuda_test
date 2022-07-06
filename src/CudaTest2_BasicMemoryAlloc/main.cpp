#include "cuda.h"
#include "nvml.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "gpu_global.cuh"

#define BUFFER_SIZE 1048576 * 16

int main()
{
    //create context automatically
    cudaError_t cuda_err = cudaSetDevice(0);
    void* dev_buf = NULL;
    void* host_buf = NULL;
    //alloc device memory
    cuda_err = cudaMalloc(&dev_buf, BUFFER_SIZE);
    //alloc nonpagedpool in host.
    cuda_err = cudaMallocHost(&host_buf, BUFFER_SIZE);
    CUDA_POINTER_ATTRIBUTE_P2P_TOKENS tokens = {0};
    //CUresult result = cuPointerGetAttribute(&tokens, CU_POINTER_ATTRIBUTE_P2P_TOKENS, ptr);
    cudaPointerAttributes attr;
    cuda_err = cudaPointerGetAttributes(&attr, dev_buf);
    cuda_err = cudaPointerGetAttributes(&attr, host_buf);
    //call my gpu kernel , device threads are launched in gpu kernel function.

    cudaFree(dev_buf);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cuda_err = cudaDeviceReset();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


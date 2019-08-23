
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256
#define ARRAY_SIZE 16777216

typedef struct timeval tval;

/**
 * Helper method to generate a very naive "hash".
 */
float generate_hash(int n, float *y)
{
    float hash = 0.0f;
    
    for (int i = 0; i < n; i++)
    {
        hash += y[i];
    }
    
    return hash;
}

/**
 * Helper method that calculates the elapsed time between two time intervals (in milliseconds).
 */
long get_elapsed(tval t0, tval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000;
}

/**
 * SAXPY reference implementation using the CPU.
 */
void cpu_saxpy(int n, float a, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = a * x[i] + y[i];
    }
}

////////////////
// TO-DO #2.6 /////////////////////////////////////////////////////////////
// Declare the kernel gpu_saxpy() with the same interface as cpu_saxpy() //
///////////////////////////////////////////////////////////////////////////

__global__ void gpu_saxpy(int n,float a,float *x,float *y){
    int idx = threadIdx.x+blockIdx.x * blockDim.x;

    y[idx] = a * x[idx] + y[idx];    

}


void test_gpu_saxpy_stream(){
    float a     = 1.0f;
    float *x    = NULL;
    float *y    = NULL;
    float error = 0.0f;

    x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    y = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    
    // Initialize them with fixed values
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        x[i] = 0.1f;
        y[i] = 0.2f;
    }


    float *d_x,*d_y;
    cudaMalloc(&d_x,ARRAY_SIZE*sizeof(float));
    cudaMalloc(&d_y,ARRAY_SIZE*sizeof(float));

    // cudaMemcpy(d_x,x,ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice);
    // cudaMemcpy(d_y,y,ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice);

    cudaStream_t stream1,stream2;
    cudaError_t result1 = cudaStreamCreate(&stream1);
    cudaError_t result2 = cudaStreamCreate(&stream2);

    cudaMemcpyAsync(d_x,x,ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice,stream1);
    cudaMemcpyAsync(d_y,y,ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice,stream2);

    dim3 grid(ARRAY_SIZE/512);     // 1 block in the grid
    dim3 block(512);   // 32 threads per block

    // cudaDeviceSynchronize();
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    gpu_saxpy<<<grid,block>>>(ARRAY_SIZE,a,d_x,d_y);

    cudaMemcpy(y,d_y,ARRAY_SIZE*sizeof(float),cudaMemcpyDeviceToHost);

    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    // Calculate the "hash" of the result from the CPU
    error = generate_hash(ARRAY_SIZE, y);
    

    // Calculate the "hash" of the result from the GPU
    error = fabsf(error - generate_hash(ARRAY_SIZE, y));
    
    // Confirm that the execution has finished
    printf("Execution finished (error=%.6f).\n", error);
    
    if (error > 0.0001f)
    {
        fprintf(stderr, "Error: The solution is incorrect!\n");
    }
    
    // Release all the allocations
    free(x);
    free(y);
    
}


int main(int argc, char **argv)
{
    float a     = 0.0f;
    float *x    = NULL;
    float *y    = NULL;
    float error = 0.0f;
    
    ////////////////
    // TO-DO #2.2 ///////////////////////////////
    // Introduce the grid and block definition //
    /////////////////////////////////////////////
    
    //////////////////
    // TO-DO #2.3.1 /////////////////////////////
    // Declare the device pointers d_x and d_y //
    /////////////////////////////////////////////
    
    // Make sure the constant is provided
    if (argc != 2)
    {
        fprintf(stderr, "Error: The constant is missing!\n");
        return -1;
    }
    
    // Retrieve the constant and allocate the arrays on the CPU
    a = atof(argv[1]);
    x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    y = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    
    // Initialize them with fixed values
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        x[i] = 0.1f;
        y[i] = 0.2f;
    }
    
    //////////////////
    // TO-DO #2.3.2 ////////////////////////////////////////////////////////
    // Allocate d_x and d_y on the GPU, and copy the content from the CPU //
    ////////////////////////////////////////////////////////////////////////
    
    // Call the CPU code
    cpu_saxpy(ARRAY_SIZE, a, x, y);

    //test_gpu_saxpy_stream();

    float *d_x,*d_y;
    cudaMalloc(&d_x,ARRAY_SIZE*sizeof(float));
    cudaMalloc(&d_y,ARRAY_SIZE*sizeof(float));

    cudaMemcpy(d_x,x,ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,y,ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice);

    dim3 grid(ARRAY_SIZE/1024);     // 1 block in the grid
    dim3 block(1024);   // 32 threads per block

    gpu_saxpy<<<grid,block>>>(ARRAY_SIZE,a,d_x,d_y);

    cudaMemcpy(y,d_y,ARRAY_SIZE*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);

    
    // Calculate the "hash" of the result from the CPU
    error = generate_hash(ARRAY_SIZE, y);
    
    ////////////////
    // TO-DO #2.4 ////////////////////////////////////////
    // Call the GPU kernel gpu_saxpy() with d_x and d_y //
    //////////////////////////////////////////////////////
    
    //////////////////
    // TO-DO #2.5.1 ////////////////////////////////////////////////////
    // Copy the content of d_y from the GPU to the array y on the CPU //
    ////////////////////////////////////////////////////////////////////
    
    // Calculate the "hash" of the result from the GPU
    error = fabsf(error - generate_hash(ARRAY_SIZE, y));
    
    // Confirm that the execution has finished
    printf("Execution finished (error=%.6f).\n", error);
    
    if (error > 0.0001f)
    {
        fprintf(stderr, "Error: The solution is incorrect!\n");
    }
    
    // Release all the allocations
    free(x);
    free(y);
    
    //////////////////
    // TO-DO #2.5.2 /////////
    // Release d_x and d_y //
    /////////////////////////
    
    return 0;
}


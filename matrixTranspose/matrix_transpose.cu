#include "define.h"

#define BDIMY 32
#define BDIMX 16

__global__ void transposeSmem(float *in,float *out,int nx,int ny){
    //static shared memory
    __shared__ float tile[BDIMY][BDIMX];

    //coordinate of original matrix
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    //linear global memory index for original matrix
    unsigned int ti = iy*nx + ix;

    //thread index in transpose block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx/blockDim.y;
    unsigned int icol = bidx%blockDim.y;

    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;

    unsigned int to = iy * ny + ix;
    
    if(ix<nx && iy<ny){
        //load data to shared memory from global memory
        tile[threadIdx.y][threadIdx.x] = in[ti];

        //sync all the threads inside the block
        __syncthreads();

        //store data to global memory from shared memory
        out[to]=tile[icol][irow];
    }

}

int main(){
    const int nx = 1<<5,ny=1<<5;
    float *in = new float[nx*ny];
    float *out = new float[nx*ny];
   
    for(auto i=in;i<in+nx*ny;++i){
         *i = rand()%9;
    }
    
    float *d_in,*d_out;
    gpuErrchk(cudaMalloc(&d_in,nx*ny*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_out,nx*ny*sizeof(float)));

    gpuErrchk(cudaMemcpy(d_in,in,nx*ny*sizeof(float),cudaMemcpyHostToDevice));

    dim3 threadPerBlock(16,16);
    dim3 block((nx+threadPerBlock.x-1)/threadPerBlock.x,(ny+threadPerBlock.y-1)/threadPerBlock.y);
    transposeSmem<<<block,threadPerBlock>>>(d_in,d_out,nx,ny);
    gpuErrchk(cudaMemcpy(out,d_out,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));


    for(auto i=in;i<in+nx*ny;++i){
        if((i-in)%nx==0){
            printf("\n");    
        }
        printf("%.1f ",*i);
    }

    printf("\n\n");

    for(auto i=out;i<out+nx*ny;++i){
        if((i-out)%ny==0){
            printf("\n");    
        }
        printf("%.1f ",*i);
    }
    printf("\n");

    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));
    delete[]in;
    delete[]out;

    return 0;
}
#include "define.h"
#include <iostream>

__global__ void addVec(float *a,float *b,float *c,int wd,int ht){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int index = idx*ht+idy;
    c[index]=a[index]+b[index];
}


void test(){
    int wd=10,ht=10;
 
    float *a,*b,*c;
    float *d_a,*d_b,*d_c;
    gpuErrchk(cudaHostAlloc(&a,wd*ht*sizeof(float),cudaHostAllocMapped));
    gpuErrchk(cudaHostAlloc(&b,wd*ht*sizeof(float),cudaHostAllocMapped));
    gpuErrchk(cudaHostAlloc(&c,wd*ht*sizeof(float),cudaHostAllocMapped));

    cudaHostGetDevicePointer(&d_a,a,0);
    cudaHostGetDevicePointer(&d_b,b,0);
    cudaHostGetDevicePointer(&d_c,c,0);

    for(int i=0;i<wd*ht;++i){
        a[i]=i;
        b[i]=10;
    }

    dim3 block(2,2),t(5,5);
    addVec<<<block,t>>>(d_a,d_b,d_c,wd,ht);

    for(int i=0;i<wd;++i){
        for(int j=0;j<ht;++j){
            std::cout<<c[i*ht+j]<<" ";
        }
        std::cout<<std::endl;
    }
}

int main(){
    test();

    return 0;
}
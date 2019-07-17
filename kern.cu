#include "define.h"
#include <iostream>

__global__ void addVec(float *a,float *b,float *c,int wd,int ht){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // int idy = blockDim.y * blockIdx.y + threadIdx.y;

    // int index = idx*ht+idy;
    
    c[idx]=b[idx];//+b[index];
}


void test(){
    int wd=10,ht=10;
 
    float *a,*b,*c;
    float *d_a,*d_b,*d_c;
    //gpuErrchk(cudaHostAlloc(&a,wd*ht*sizeof(float),cudaHostAllocMapped | cudaHostAllocWriteCombined));
    gpuErrchk(cudaHostAlloc(&b,wd*ht*sizeof(float),cudaHostAllocMapped | cudaHostAllocWriteCombined));
    gpuErrchk(cudaHostAlloc(&c,wd*ht*sizeof(float),cudaHostAllocMapped));

    //cudaHostGetDevicePointer(&d_a,a,0);
    cudaHostGetDevicePointer(&d_b,b,0);
    cudaHostGetDevicePointer(&d_c,c,0);

    for(int i=0;i<wd*ht;++i){
        //std::cout<<i<<" ";
        a[i]=1;
        b[i]=10;
    }
    //std::cout<<std::endl;

    //dim3 block(5,5),t(2,2);
    addVec<<<10,10>>>(d_a,d_b,d_c,wd,ht);

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
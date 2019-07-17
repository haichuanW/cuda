#pragma once
//define.h
//定义线程数、debug相关
//定义最大连续合成张数、小图缩小倍数
#include<cuda_runtime.h>
#include "cuda.h"
#include "device_functions.h"
#include<stdio.h>
#include<stdlib.h>
#include<cufft.h>

#ifdef __INTELLISENSE__
	#ifndef __CUDACC__ 
		#define __CUDACC__
	#endif
	#include <device_functions.h>
	#include <device_launch_parameters.h>
	#include <cuda_texture_types.h>
#endif

#define FASTSPLINE//修改后样条你和，适用于间隔均匀且连续的样本点

#define THREADNUM 4//核心数量
#define RESIZEXY 4//缩小图像倍数
#define MAXCONTALIGNNUM 8//最大连续合成图像数量
//#define CUDADEBUG
#define CUDAMALLOC

#ifdef CUDAMALLOC

#define HOSTMALLOC(point,size,type) cudaMallocHost((void**)&(point),(size)*sizeof(type))//申请页锁定内存
#define HOSTFREE(point) cudaFreeHost(point)
#else
#define HOSTMALLOC(point,size,type) (point)=new type[(size)]
#define HOSTFREE(point) free(point)
#endif
//debug
#ifdef CUDADEBUG
#define KNERR() cudaGetLastError()
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
static const char *cufftGetErrorString(cufftResult error)
{
	switch (error)
	{
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	}

	return "<unknown>";
}
inline void gpuAssert(cufftResult code, const char *file, int line, bool abort = true)
{
	if (code != CUFFT_SUCCESS)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cufftGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#else
#define gpuErrchk(ans) ans
#define KNERR() 
#endif
//lib
#ifdef IMGPROHD_EXPORTS
#define IMGPRO_API __declspec(dllexport)
#else
#define IMGPRO_API 
#endif
//block

#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_

extern "C++" int32_t cuda_init(void);

extern "C++" bool cuda_raw2gray(int width, int height, unsigned char *img,
                                unsigned char *res);

#endif

//
// Created by birdpeople on Fri Aug 04 2023, 16:14:42
// Copyright (c) 2023
//

#pragma once
#ifndef TYPE_DEF_H
#define TYPE_DEF_H
namespace knn {
    #ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
    #define DEVICE_CALLABLE __device__
    #define HOST_CALLABLE __host__
    #define DEVICE_INLINE_CALLABLE __device__ __forceinline__
    #define HOST_INLINE_CALLABLE __host__ __forceinline__
    #define CUDA_INLINE_CALLABLE __host__ __device__ __forceinline__
    #define INLINE_CALLABLE __forceinline__
    #else
    #define CUDA_CALLABLE
    #define DEVICE_CALLABLE
    #define HOST_CALLABLE
    #define DEVICE_INLINE_CALLABLE
    #define CUDA_INLINE_CALLABLE inline
    #define HOST_INLINE_CALLABLE inline
    #define INLINE_CALLABLE inline
    #endif
}
#endif // TYPE_DEF_H

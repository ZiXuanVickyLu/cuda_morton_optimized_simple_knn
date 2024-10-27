//
// Created by birdpeople on 8/9/2023.
//

#ifndef AABB_H
#define AABB_H

#include "vector_type_t.h"

namespace knn{

    template<typename T>
    auto constexpr infinity_AABB(){return static_cast<T>(1e6);}

    template<typename T = float>
    struct AABB{
        using value_type = T;
        using point = float3;

        point minP;
        point maxP;

        //a initialized but not assigned AABB is invalid, which will return non-overlap with any other AABB
        CUDA_INLINE_CALLABLE AABB() : minP(make_float3(infinity_AABB<T>())), maxP(make_float3(-infinity_AABB<T>())) {}

        CUDA_INLINE_CALLABLE AABB(const point &_minP, const point &_maxP) :
                minP(_minP), maxP(_maxP) {}

        CUDA_INLINE_CALLABLE AABB &unify(const point &p) {
            minP = min(minP, p);
            maxP = max(maxP, p);
            return *this;
        }

        CUDA_INLINE_CALLABLE AABB &dilate(T thickness) {
            auto scale_length = make_float3(thickness, thickness, thickness);
            minP -= scale_length;
            maxP += scale_length;
            if(thickness < 0){
                auto minP_ = minP; auto maxP_ = maxP;
                minP = make_float3(min(minP_.x, maxP_.x), min(minP_.y, maxP_.y), min(minP_.z, maxP_.z));
                maxP = make_float3(max(minP_.x, maxP_.x), max(minP_.y, maxP_.y), max(minP_.z, maxP_.z));
            }
            return *this;
        }

        CUDA_INLINE_CALLABLE T calc_half_surface_area() const {
            point d = maxP - minP;
            return d.x * d.y + d.y * d.z + d.z * d.x;
        }

        //allow equal, equal is not overlap
        CUDA_INLINE_CALLABLE bool is_overlap(const AABB<T>& aabb) const{
            return aabb.is_valid() && this->is_valid() &&
            !((minP.x >= aabb.maxP.x || minP.y >= aabb.maxP.y || minP.z >= aabb.maxP.z) ||
              (maxP.x <= aabb.minP.x || maxP.y <= aabb.minP.y || maxP.z <= aabb.minP.z));

        }

        CUDA_INLINE_CALLABLE AABB merge(const AABB &v) const {
            auto p_min = make_float3(min(minP.x, v.minP.x), min(minP.y, v.minP.y), min(minP.z, v.minP.z));
            auto p_max = make_float3(max(maxP.x, v.maxP.x), max(maxP.y, v.maxP.y), max(maxP.z, v.maxP.z));
            return {p_min, p_max};
        }

        CUDA_INLINE_CALLABLE bool is_valid() const{
            return this->minP.x != infinity_AABB<T>() && this->minP.y != infinity_AABB<T>() && this->minP.z != infinity_AABB<T>() &&
                   this->maxP.x != -infinity_AABB<T>() && this->maxP.y != -infinity_AABB<T>() && this->maxP.z != -infinity_AABB<T>() &&
                   this->minP.x <= this->maxP.x && this->minP.y <= this->maxP.y && this->minP.z <= this->maxP.z;
        }

    };

    template<>
    struct AABB<double>{
        using value_type = double;
        using point = double3;

        point minP;
        point maxP;

        //a initialized but not assigned AABB is invalid, which will return non-overlap with any other AABB
        CUDA_INLINE_CALLABLE AABB() : minP(make_double3(infinity_AABB<double>())), maxP(make_double3(-infinity_AABB<double>())) {}

        CUDA_INLINE_CALLABLE AABB(const point &_minP, const point &_maxP) :
                minP(_minP), maxP(_maxP) {}

        CUDA_INLINE_CALLABLE AABB &unify(const point &p) {
            minP = min(minP, p);
            maxP = max(maxP, p);
            return *this;
        }

        CUDA_INLINE_CALLABLE AABB &dilate(double thickness) {
            auto scale_length = make_double3(thickness, thickness, thickness);
               minP -= scale_length;
               maxP += scale_length;
               if(thickness < 0){
                   auto minP_ = minP; auto maxP_ = maxP;
                   minP = make_double3(min(minP_.x, maxP_.x), min(minP_.y, maxP_.y), min(minP_.z, maxP_.z));
                   maxP = make_double3(max(minP_.x, maxP_.x), max(minP_.y, maxP_.y), max(minP_.z, maxP_.z));
               }
            return *this;
        }

        CUDA_INLINE_CALLABLE double calc_half_surface_area() const {
            point d = maxP - minP;
            return d.x * d.y + d.y * d.z + d.z * d.x;
        }

        //allow equal, equal is not overlap
        CUDA_INLINE_CALLABLE bool is_overlap(const AABB<double>& aabb) const{
            return  aabb.is_valid() && this->is_valid() &&
            !((minP.x >= aabb.maxP.x || minP.y >= aabb.maxP.y || minP.z >= aabb.maxP.z) ||
                      (maxP.x <= aabb.minP.x || maxP.y <= aabb.minP.y || maxP.z <= aabb.minP.z));
        }

        CUDA_INLINE_CALLABLE AABB merge(const AABB &v) const {
            auto p_min = make_double3(min(minP.x, v.minP.x), min(minP.y, v.minP.y), min(minP.z, v.minP.z));
            auto p_max = make_double3(max(maxP.x, v.maxP.x), max(maxP.y, v.maxP.y), max(maxP.z, v.maxP.z));
            return {p_min, p_max};
        }

        CUDA_INLINE_CALLABLE bool is_valid() const{
            return this->minP.x != infinity_AABB<double>() && this->minP.y != infinity_AABB<double>() && this->minP.z != infinity_AABB<double>() &&
                   this->maxP.x != -infinity_AABB<double>() && this->maxP.y != -infinity_AABB<double>() && this->maxP.z != -infinity_AABB<double>() &&
                   this->minP.x <= this->maxP.x && this->minP.y <= this->maxP.y && this->minP.z <= this->maxP.z;
        }

    };

    template<typename T>
    CUDA_INLINE_CALLABLE bool is_valid(const AABB<T>& aabb) {
        return aabb.minP.x != infinity_AABB<T>() && aabb.minP.y != infinity_AABB<T>() && aabb.minP.z != infinity_AABB<T>() &&
               aabb.maxP.x != -infinity_AABB<T>() && aabb.maxP.y != -infinity_AABB<T>() && aabb.maxP.z != -infinity_AABB<T>() &&
               aabb.minP.x <= aabb.maxP.x && aabb.minP.y <= aabb.maxP.y && aabb.minP.z <= aabb.maxP.z;
    }

template<typename T= float>
    struct AABBAsOrderedInteger {
        using order_type = int32_t;

        float3AsOrderedInt minP;
        float3AsOrderedInt maxP;

        CUDA_INLINE_CALLABLE AABBAsOrderedInteger() :
                minP(make_float3(infinity_AABB<T>())), maxP(make_float3(-infinity_AABB<T>())) {
        }
        CUDA_INLINE_CALLABLE explicit AABBAsOrderedInteger(const AABB<T> &v) :
                minP(v.minP), maxP(v.maxP) {
        }

        AABBAsOrderedInteger& operator= (const AABBAsOrderedInteger &v) = default;

        CUDA_INLINE_CALLABLE explicit operator AABB<T>() const {
            return AABB(static_cast<float3>(minP), static_cast<float3>(maxP));
        }

    };


template<>
    struct AABBAsOrderedInteger<double> {
        double3AsOrderedLongLong minP;
        double3AsOrderedLongLong maxP;

        CUDA_INLINE_CALLABLE AABBAsOrderedInteger() :
                minP(make_double3(infinity_AABB<double>())), maxP(make_double3(-infinity_AABB<double>())) {
        }
        CUDA_INLINE_CALLABLE explicit AABBAsOrderedInteger(const AABB<double> &v) :
                minP(v.minP), maxP(v.maxP) {
        }

        AABBAsOrderedInteger& operator= (const AABBAsOrderedInteger &v) = default;

        CUDA_INLINE_CALLABLE explicit operator AABB<double>() const {
            return {static_cast<double3>(minP), static_cast<double3>(maxP)};
        }
    };

}

#endif //AABB_H

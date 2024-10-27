//
// Created by birdpeople on 8/8/2023.
//
#ifndef GEOMETRY_H
#define GEOMETRY_H
#include "typedef.h"
#include "vector_type_t.h"

    namespace knn{

        template<typename T = float>
        struct GeometryTraits{
            using value_type = T;
            using index_type = unsigned int; //notice, since uint1 is not supported cmp operator
            using point_type = float3;
            template<typename ...Args>
            CUDA_INLINE_CALLABLE static point_type make_point(Args ...args);
            template<typename ...Args>
            CUDA_INLINE_CALLABLE static index_type make_index(Args ...args);
        };

        //bug of msvc using alias template
        //see https://stackoverflow.com/questions/49521073/msvcerror-c2244-unable-to-match-function-definition-to-an-existing-declaratio


        template<typename T>
        template<typename ...Args>
        CUDA_INLINE_CALLABLE typename GeometryTraits<T>::point_type GeometryTraits<T>::make_point(Args ...args)
        { return make_float3(std::forward<Args>(args)...);}

        template<typename T>
        template<typename ...Args>
        CUDA_INLINE_CALLABLE  typename GeometryTraits<T>::index_type GeometryTraits<T>::make_index(Args ...args)
        { return make_uint(std::forward<Args>(args)...);}

        template<>
        struct GeometryTraits<double>{
            using value_type = double;
            using index_type = unsigned int; //notice, since uint1 is not supported cmp operator
            using point_type = double3;

            template<typename ...Args>
            CUDA_INLINE_CALLABLE static point_type make_point(Args ...args);

            template<typename ...Args>
            CUDA_INLINE_CALLABLE static index_type make_index(Args ...args);
        };


        template<typename ...Args>
        CUDA_INLINE_CALLABLE typename GeometryTraits<double>::point_type GeometryTraits<double>::make_point(Args ...args)
        { return make_double3(std::forward<Args>(args)...);}

        template<typename ...Args>
        CUDA_INLINE_CALLABLE typename GeometryTraits<double>::index_type GeometryTraits<double>::make_index(Args ...args)
        { return make_uint(std::forward<Args>(args)...);}

        template<typename T = float>
        using geo_t = GeometryTraits<T>;
        using geo_f = GeometryTraits<float>;
        using geo_d = GeometryTraits<double>;

    }


#endif //GEOMETRY_H
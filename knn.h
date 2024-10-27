//
// Created by birdpeople on 5/17/2024.
//

#ifndef KNN_H
#define KNN_H
#include "geometry.h"
#include <thrust/device_vector.h>
#include "aabb.h"
//As we tested, this is the best block size for knn (minimal grid size).
constexpr size_t knn_block_size = 32;

constexpr unsigned int invalid_index = 0xffffffff;

//exact knn with rejection distance
//one can avoid to set the rejection distance (will be set to 1e10) to obtain exact knn
namespace knn{
    template<typename T, unsigned int k = 4>
    struct KNN{
        using point = typename geo_t<T>::point_type;
        using index = typename geo_t<T>::index_type;
        static constexpr unsigned int k_rank = k;
        static constexpr index dummy_idx = 0xffffffff;
        void knn(const point* refs, const point* queries, unsigned int ref_size, unsigned int query_size);
        [[nodiscard]] inline bool is_valid() const { return m_is_valid; }
        void destroy();
        index* pt_index{nullptr};              //query * k
        T* pt_distance_square{nullptr};        //query * k

        private:
        index* sequence_indices{nullptr};      //ref size
        index* sorted_indices{nullptr};        //ref size
        index* morton_codes{nullptr};          //ref size

        index* sequence_indices_query{nullptr};//query size
        index* sorted_indices_query{nullptr};  //query size
        index* morton_codes_query{nullptr};    //query size
        AABB<T>* aabbs{nullptr};               //ref size/knn_block_size = hash grid size
        void realloc(size_t query_size);
        void realloc_ref(size_t ref_size);
        //for cub radix sort and cub reduce
        size_t temp_storage_bytes{0};
        std::shared_ptr<thrust::device_vector<index>> cub_temp_storge{nullptr}; //temp_storge_size
        bool m_is_valid = false;
        unsigned int current_query_size{0};
        unsigned int current_ref_size{0};
    };
}

#include "knn.inl"
#endif //KNN_H

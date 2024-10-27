#include "knn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <cooperative_groups.h>
#include "typedef.h"
#include "aabb.h"

namespace cg = cooperative_groups;
using namespace knn;
template<typename T>
constexpr T infinity_distance() { return static_cast<T>(1e6); }

template<typename pt>
struct CustomMin
{
    DEVICE_INLINE_CALLABLE
        pt operator()(const pt& a, const pt& b) const {
        return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
    }
};

template<typename pt>
struct CustomMax
{
    DEVICE_INLINE_CALLABLE
        pt operator()(const pt& a, const pt& b) const {
        return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
    }
};

template<int K, typename pt, typename index, typename T>
DEVICE_INLINE_CALLABLE void update_K_best(const pt& ref, const pt& point, const index& pid, T* k_distance_sqr, index* k_idx)
{
    pt d = ref - point;
    T dist_sqr = d.x * d.x + d.y * d.y + d.z * d.z;
    index idx = pid;
    T t; index i;
#pragma unroll K
    for (int j = 0; j < K; j++)
    {
        if (k_distance_sqr[j] > dist_sqr)
        {
        //swap and insertion sort
            t = k_distance_sqr[j];
            k_distance_sqr[j] = dist_sqr;
            dist_sqr = t;

            i = k_idx[j];
            k_idx[j] = idx;
            idx = i;
        }
    }
}

CUDA_INLINE_CALLABLE uint32_t prepMorton(uint32_t x)
{
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    return x;
}

template<typename pt>
CUDA_INLINE_CALLABLE uint32_t coord2Morton(pt coord, pt minn, pt maxx)
{
    uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
    uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
    uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

    return x | (y << 1) | (z << 2);
}

template<typename pt, typename index>
__global__ void coord2Morton(const pt *ref, index* mt_code, pt minn, pt maxx, unsigned int size_)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= size_)
        return;

    mt_code[idx] = coord2Morton(ref[idx], minn, maxx);
}

template<typename T, typename pt>
CUDA_INLINE_CALLABLE T dist_sqr_box_point(const AABB<T>& box, const pt& p)
{
    pt diff = geo_t<T>::make_point(0);
    if (p.x < box.minP.x || p.x > box.maxP.x)
        diff.x = min(abs(p.x - box.minP.x), abs(p.x - box.maxP.x));
    if (p.y < box.minP.y || p.y > box.maxP.y)
        diff.y = min(abs(p.y - box.minP.y), abs(p.y - box.maxP.y));
    if (p.z < box.minP.z || p.z > box.maxP.z)
        diff.z = min(abs(p.z - box.minP.z), abs(p.z - box.maxP.z));
    return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}


template<typename point, typename T, typename index>
__global__ void  build_hash_grid(point* ref, index* sorted_idx, AABB<T>* aabbs, unsigned int ref_size)
{
    auto idx = cg::this_grid().thread_rank();

    AABB<T> me{};
    auto point_size  = ref_size;
    if (idx < point_size)
    {
        me.minP = ref[sorted_idx[idx]];
        me.maxP = ref[sorted_idx[idx]];
    }

    __shared__  T result_min_x[knn_block_size];
    __shared__  T result_min_y[knn_block_size];
    __shared__  T result_min_z[knn_block_size];
    __shared__  T result_max_x[knn_block_size];
    __shared__  T result_max_y[knn_block_size];
    __shared__  T result_max_z[knn_block_size];

#pragma unroll 10
    for (int off = knn_block_size / 2; off >= 1; off /= 2)
    {
        if (threadIdx.x < 2 * off){
            result_min_x[threadIdx.x] = me.minP.x;
            result_min_y[threadIdx.x] = me.minP.y;
            result_min_z[threadIdx.x] = me.minP.z;
            result_max_x[threadIdx.x] = me.maxP.x;
            result_max_y[threadIdx.x] = me.maxP.y;
            result_max_z[threadIdx.x] = me.maxP.z;
        }
        __syncthreads();

        if (threadIdx.x < off)
        {
            me.minP.x = min(me.minP.x, result_min_x[threadIdx.x + off]);
            me.minP.y = min(me.minP.y, result_min_y[threadIdx.x + off]);
            me.minP.z = min(me.minP.z, result_min_z[threadIdx.x + off]);
            me.maxP.x = max(me.maxP.x, result_max_x[threadIdx.x + off]);
            me.maxP.y = max(me.maxP.y, result_max_y[threadIdx.x + off]);
            me.maxP.z = max(me.maxP.z, result_max_z[threadIdx.x + off]);

        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
       aabbs[blockIdx.x] = me;
}

template< typename pt, typename T, typename index, unsigned int K>
__global__ void knn_search(const pt* refs,
    const pt* queries,
    const index* sorted_query_idx,
    const index* sorted_ref_idx,
    const AABB<T>* aabbs, T* k_distance_sqr,
    index* k_idx, unsigned int ref_size, unsigned int queries_size)
{
    int idx = cg::this_grid().thread_rank();
    if (idx >= queries_size)
        return;
    index pid = sorted_query_idx[idx];
    T* best_distance_sqr = k_distance_sqr+ pid * K;
    index* best_idx = k_idx + pid * K;
    pt point = queries[pid];

    int i;
#pragma unroll K
    for(i = 0; i< K; i++){
        best_distance_sqr[i] = infinity_distance<T>();
        best_idx[i] = invalid_index;
    }

AABB<T> box; T dist_sqr;
    for (int gid = 0; gid < (ref_size + knn_block_size - 1) / knn_block_size; ++gid)
    {
         box = aabbs[gid];
         dist_sqr = dist_sqr_box_point(box, point);
         if ( dist_sqr > best_distance_sqr[K - 1])
            continue;

#pragma unroll knn_block_size
        for (i = gid * knn_block_size; i < min(ref_size, unsigned int( (gid + 1) * knn_block_size)); i++)
        {
            update_K_best<K, pt, index, T>(refs[sorted_ref_idx[i]], point, sorted_ref_idx[i], best_distance_sqr, best_idx);
        }
    }

}


namespace knn{
    template<typename T, unsigned int k>
    void KNN<T, k>::realloc(size_t query_size){
        if(pt_index == nullptr || current_query_size != query_size)
        {
            current_query_size = query_size;
            if(pt_index != nullptr)
            {
                //free
                cudaFree(pt_index); pt_index = nullptr;
                cudaFree(pt_distance_square); pt_distance_square = nullptr;
                cudaFree(sequence_indices_query); sequence_indices_query = nullptr;
                cudaFree(sorted_indices_query); sorted_indices_query = nullptr;
                cudaFree(morton_codes_query); morton_codes_query = nullptr;
            }
            cudaMalloc((void **) &pt_index, query_size * k * sizeof(index));
            cudaMemset(pt_index, dummy_idx, query_size * k * sizeof(index));
            cudaMalloc((void **) &pt_distance_square, query_size * k * sizeof(T));
            cudaMemset(pt_distance_square, infinity_distance<T>(), query_size * k * sizeof(T));
            cudaMalloc((void**) &sequence_indices_query, query_size * sizeof(index));
            thrust::device_ptr<index> sequence_indices_query_ptr(sequence_indices_query);
            thrust::sequence(sequence_indices_query_ptr, sequence_indices_query_ptr + query_size);
            cudaMalloc((void**) &sorted_indices_query, query_size * sizeof(index));
            cudaMalloc((void**) &morton_codes_query, query_size * sizeof(index));
        }else{
            //make initialize
            cudaMemset(pt_index, dummy_idx, query_size * k * sizeof(index));
            cudaMemset(pt_distance_square, infinity_distance<T>(), query_size * k * sizeof(T));
        }
    }

    template<typename T, unsigned int k>
    void KNN<T, k>::realloc_ref(size_t ref_size){
        if(sequence_indices == nullptr || current_ref_size != ref_size)
        {
            current_ref_size = ref_size;
            if(sequence_indices!= nullptr)
            {
                //free
                cudaFree(sequence_indices); sequence_indices = nullptr;
                cudaFree(sorted_indices); sorted_indices = nullptr;
                cudaFree(morton_codes); morton_codes = nullptr;
                cudaFree(aabbs); aabbs = nullptr;
            }
            cudaMalloc((void**) &sequence_indices, ref_size * sizeof(index));
            thrust::device_ptr<index> sequence_indices_ptr(sequence_indices);
            thrust::sequence(sequence_indices_ptr, sequence_indices_ptr + ref_size);
            cudaMalloc((void**) &sorted_indices, ref_size * sizeof(index));

            cudaMalloc((void**) &morton_codes, ref_size * sizeof(index));
            cudaMalloc((void**) &aabbs, ((ref_size + knn_block_size - 1) / knn_block_size) * sizeof(AABB<T>));
            thrust::device_ptr<AABB<T>> aabbs_ptr(aabbs);
            thrust::fill(aabbs_ptr, aabbs_ptr + ((ref_size + knn_block_size - 1) / knn_block_size), AABB<T>());
        }else{
            //no initialization need.

        }
        if(cub_temp_storge.get()==nullptr){
            cub_temp_storge = std::make_shared<thrust::device_vector<index>>(ref_size * 4);
        }
    }

    //build hash_grid
    //hash code (hash grid cluster idx): morton code % knn_block_size
    template<typename T, unsigned int k>
    void KNN<T, k>::knn(const point* refs, const point* queries, unsigned int ref_size, unsigned int query_size)
    {
        cudaEvent_t start, stop;

        cudaEventCreate(&start);  cudaEventCreate(&stop);
        cudaEventRecord(start);
        realloc(query_size);
        realloc_ref(ref_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("realloc time (ms): %f\n", milliseconds);

        cudaEventRecord(start);
        point* result;
        cudaMalloc(&result, sizeof(point));
        point init = geo_t<T>::make_point(0, 0, 0);
        point minn,maxx;
        cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, refs, result, ref_size, CustomMin<point>(), init);
        if (temp_storage_bytes > 0 && cub_temp_storge->size() < temp_storage_bytes) {
            cub_temp_storge->resize(temp_storage_bytes);
        }
        cub::DeviceReduce::Reduce(cub_temp_storge->data().get(), temp_storage_bytes, refs, result, ref_size, CustomMin<point>(), init);
        cudaMemcpy(&minn, result, sizeof(point), cudaMemcpyDeviceToHost);

        cub::DeviceReduce::Reduce(cub_temp_storge->data().get(), temp_storage_bytes, refs, result, ref_size, CustomMax<point>(), init);
        cudaMemcpy(&maxx, result, sizeof(point), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("reduce time (ms): %f\n", milliseconds);

        cudaEventRecord(start);
        coord2Morton<point, index> << <(ref_size + 255) / 256, 256 >> > (refs, morton_codes, minn, maxx, ref_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("coord2Morton ref time (ms): %f\n", milliseconds);

        cudaEventRecord(start);
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton_codes, morton_codes, sequence_indices, sorted_indices, ref_size);
        if(temp_storage_bytes > 0 && cub_temp_storge->size() < temp_storage_bytes){
            cub_temp_storge->resize(temp_storage_bytes);
        }
        cub::DeviceRadixSort::SortPairs(cub_temp_storge->data().get(), temp_storage_bytes, morton_codes, morton_codes, sequence_indices, sorted_indices, ref_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("sort ref time (ms): %f\n", milliseconds);

        cudaEventRecord(start);
        auto number_hash_grids = (ref_size + knn_block_size - 1) / knn_block_size;
        build_hash_grid<< <number_hash_grids, knn_block_size >> > (refs, sorted_indices, aabbs, ref_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("build hash grid (ms): %f\n", milliseconds);


        //build query morton code
        cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, queries, result, query_size, CustomMin<point>(), init);
        if (temp_storage_bytes > 0 && cub_temp_storge->size() < temp_storage_bytes) {
            cub_temp_storge->resize(temp_storage_bytes);
        }
        cub::DeviceReduce::Reduce(cub_temp_storge->data().get(), temp_storage_bytes, queries, result, query_size, CustomMin<point>(), init);
        cudaMemcpy(&minn, result, sizeof(point), cudaMemcpyDeviceToHost);

        cub::DeviceReduce::Reduce(cub_temp_storge->data().get(), temp_storage_bytes, queries, result, query_size, CustomMax<point>(), init);
        cudaMemcpy(&maxx, result, sizeof(point), cudaMemcpyDeviceToHost);
        coord2Morton << <(query_size + 255) / 256, 256 >> > (queries, morton_codes_query, minn, maxx, query_size);
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton_codes_query, morton_codes_query, sequence_indices_query, sorted_indices_query, query_size);
        if(temp_storage_bytes > 0 && cub_temp_storge->size() < temp_storage_bytes){
            cub_temp_storge->resize(temp_storage_bytes);
        }
        cub::DeviceRadixSort::SortPairs(cub_temp_storge->data().get(), temp_storage_bytes, morton_codes_query, morton_codes_query, sequence_indices_query, sorted_indices_query, query_size);

        cudaEventRecord(start);
        knn_search<point, T,index,k> << <(query_size + 128 - 1) / 128, 128 >> > (
            refs,
            queries,
            sorted_indices_query,
            sorted_indices,
            aabbs,
            pt_distance_square,
            pt_index, ref_size, query_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("knn search time (ms): %f\n", milliseconds);
        m_is_valid = true;
    }


    template<typename T, unsigned int k>
    void KNN<T,k>::destroy(){
        if(pt_index != nullptr){
            cudaFree(pt_index); pt_index = nullptr;
            cudaFree(pt_distance_square); pt_distance_square = nullptr;
        }
        if(sequence_indices != nullptr){

            cudaFree(sequence_indices); sequence_indices = nullptr;
            cudaFree(sorted_indices); sorted_indices = nullptr;
            cudaFree(morton_codes); morton_codes = nullptr;
            cudaFree(aabbs); aabbs = nullptr;
            cudaFree(sequence_indices_query); sequence_indices_query = nullptr;
            cudaFree(sorted_indices_query); sorted_indices_query = nullptr;
            cudaFree(morton_codes_query); morton_codes_query = nullptr;

        }
        m_is_valid = false;
    }

}
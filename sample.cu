#include <iostream>
#include "knn.h"
#include<random>
using namespace knn;
#define query_size 10000
#define ref_size 1000000
#define ecc float
#define k_rank 1
int main() {
   KNN<ecc, k_rank> knn_handler;
   using point = KNN<ecc, k_rank>::point;
   using index = KNN<ecc, k_rank>::index;
   std::vector<point> ref(ref_size);
   std::vector<point> query(query_size);
   std::vector<index> indices(query_size * k_rank);
   std::vector<ecc> dists(query_size * k_rank);

   std::vector<ecc> dists_gpu(query_size * k_rank);
   std::vector<index> indices_gpu(query_size * k_rank);

   std::random_device rd;
   std::mt19937 gen(rd());

   std::uniform_real_distribution<ecc> dis(0.0, 10.0);


   ecc random_ecc = dis(gen);

   for(int i = 0; i < query_size; i++){
      query[i].x = dis(gen);
      query[i].y = dis(gen);
      query[i].z = dis(gen);
      for(int j = 0; j< k_rank; j++)
         dists[i* k_rank + j] = 1e10;
   }
   for(int i = 0; i < ref_size; i++){
      ref[i].x = dis(gen);
      ref[i].y = dis(gen);
      ref[i].z = dis(gen);
   }

   point* query_gpu;
   point* ref_gpu;
   cudaMalloc((void**) &query_gpu, query_size * sizeof(point));
   cudaMalloc((void**) &ref_gpu, ref_size * sizeof(point));
   cudaMemcpy(query_gpu, query.data(), query_size * sizeof(point), cudaMemcpyHostToDevice);
   cudaMemcpy(ref_gpu, ref.data(), ref_size * sizeof(point), cudaMemcpyHostToDevice);
   cudaDeviceSynchronize();


  // dummy case on CPU
    for(int i = 0; i< query_size; i++){
      for(int j = 0; j< ref_size; j++){
         point d = { query[i].x - ref[j].x, query[i].y - ref[j].y, query[i].z - ref[j].z };
         ecc dist = d.x * d.x + d.y * d.y + d.z * d.z;
         index idx = j;
         for (int k = 0; k < k_rank; k++)
         {
            if (dists[i*k_rank + k] > dist)
            {
               ecc t = dists[i*k_rank + k];
               dists[i*k_rank + k] = dist;
               dist = t;

               index tmp = indices[i * k_rank + k];
               indices[i*k_rank + k] = idx;
               idx = tmp;
            }
         }
      }
    }

   //you can un-comment this to see the CPU result
   std::cout << " ========================== CPU result: ============================= " << std::endl;
   // for(int i = 0; i< query_size; i++) {
   //    for(int j = 0; j< k_rank; j++) {
   //       std::cout << indices[i*k_rank + j] << " ";
   //       std::cout << dists[i*k_rank + j] << " ";
   //    }
   //    std::cout << std::endl;
   // }
   std::cout << " ========================== GPU result: ============================= " << std::endl;
   std::cout << "query size: " << query_size << " reference size: " << ref_size << std::endl;
   std::cout << "k rank: " << k_rank << std::endl;
   std::cout << "use double: " << std::is_same<ecc, double>::value << std::endl;

   knn_handler.knn(ref_gpu, query_gpu,ref_size, query_size);
   cudaMemcpy(indices_gpu.data(), knn_handler.pt_index, query_size * k_rank * sizeof(index), cudaMemcpyDeviceToHost);
   cudaMemcpy(dists_gpu.data(), knn_handler.pt_distance_square, query_size * k_rank * sizeof(ecc), cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();
   //you can un-comment this to see the GPU result
   // for(int i = 0; i< query_size; i++) {
   //    for(int j = 0; j< k_rank; j++) {
   //       std::cout << indices_gpu[i*k_rank + j] << " ";
   //       std::cout << dists_gpu[i*k_rank + j] << " ";
   //    }
   //    std::cout << std::endl;
   // }
   for(int i= 0; i< query_size; i++){
      for(int j = 0; j< k_rank; j++){
         //distance tolerance: 1e-5
         if(indices[i*k_rank + j] != indices_gpu[i*k_rank + j] && (dists[i*k_rank + j] - dists_gpu[i*k_rank + j]) > 1e-5){
            std::cout << "Error at " << i << " " << j << std::endl;
            return 1;
         }
      }
   }
   std::cout << " ========================== Test Pass! ============================== " << std::endl;
   return 0;
}


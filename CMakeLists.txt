cmake_minimum_required(VERSION 3.20)
include(cmake/detect_cuda_arch.cmake)
project(knn_sample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--heap,536870912")
detect_cuda_arch(DETECTED_CUDA_ARCH)
# Set the architecture
set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_CUDA_ARCH})
message(STATUS "CMAKE_CUDA_ARCHS = " ${CMAKE_CUDA_ARCHITECTURES})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    link_libraries(${CUDA_LIBRARIES})
else()
    message("ERROR: Cuda Not Found! [Required]")
endif()

# add thrust
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

#group up all the source files
file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.c*)
add_executable(knn_sample ${SOURCE})

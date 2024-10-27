# Function to detect CUDA architecture
function(detect_cuda_arch output_variable)
    # Temporary directory for compilation
    set(CUDA_DETECT_DIR ${CMAKE_BINARY_DIR}/cuda_detect)
    file(MAKE_DIRECTORY ${CUDA_DETECT_DIR})

    # Create a simple CUDA program to get device properties
    file(WRITE ${CUDA_DETECT_DIR}/cuda_detect.cu
            "#include <cuda_runtime.h>
        #include <stdio.h>
        int main() {
            int deviceCount;
            cudaError_t error = cudaGetDeviceCount(&deviceCount);
            if (error != cudaSuccess) {
                printf(\"CUDA_ERROR\\n\");
                return 1;
            }
            if (deviceCount == 0) {
                printf(\"NO_CUDA_DEVICES\\n\");
                return 1;
            }
            cudaDeviceProp prop;
            // Get properties of first available CUDA device
            cudaGetDeviceProperties(&prop, 0);
            // Print compute capability
            printf(\"%d%d\", prop.major, prop.minor);
            return 0;
        }")

    # Try to compile and run the detection program
    try_run(
            RUN_RESULT
            COMPILE_RESULT
            ${CUDA_DETECT_DIR}
            ${CUDA_DETECT_DIR}/cuda_detect.cu
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            RUN_OUTPUT_VARIABLE ARCH_OUTPUT
    )

    if(COMPILE_RESULT AND RUN_RESULT EQUAL 0)
        # Successfully detected architecture
        message(STATUS "[detect_cuda_arch] Detected CUDA architecture: ${ARCH_OUTPUT}")
        set(${output_variable} ${ARCH_OUTPUT} PARENT_SCOPE)
    else()
        # Fallback to a common architecture if detection fails
        message(WARNING "[detect_cuda_arch] CUDA architecture detection failed. Falling back to sm_75")
        set(${output_variable} "75" PARENT_SCOPE)
    endif()
endfunction()
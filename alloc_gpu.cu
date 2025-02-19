#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaSetDevice(0); // 选择第 0 号 GPU

    void *d_mem;
    size_t size = 7L * 1024 * 1024 * 1024;  // 2GB

    cudaError_t err = cudaMalloc(&d_mem, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Allocated 2GB on GPU, press Enter to release." << std::endl;
    std::cin.get();

    cudaFree(d_mem);
    return 0;
}

#include "dat_reader.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// 错误检查宏
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CUFFT_CHECK(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        printf("cuFFT Error at %s:%d - Code: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
}

// 类型定义
typedef float2 float2_td;
typedef cufftComplex Complex;

// 复数乘法内核（用于互相关）
__global__ void complex_multiply_kernel_td(Complex* fft1, Complex* fft2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        Complex a = fft1[idx];
        Complex b = fft2[idx];
        
        // 计算 a * conj(b)
        float real_part = a.x * b.x + a.y * b.y;
        float imag_part = a.y * b.x - a.x * b.y;
        
        fft1[idx].x = real_part;
        fft1[idx].y = imag_part;
    }
}

// 复数乘以其共轭内核（用于自相关）
__global__ void auto_correlation_kernel_td(Complex* fft, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        Complex a = fft[idx];
        
        // 计算 a * conj(a) = |a|^2
        fft[idx].x = a.x * a.x + a.y * a.y;  // 实部 = |a|^2
        fft[idx].y = 0.0f;                     // 虚部 = 0
    }
}

// 取模内核（计算幅度）
__global__ void compute_magnitude_kernel_td(Complex* complex_data, float* magnitude, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        Complex c = complex_data[idx];
        magnitude[idx] = sqrtf(c.x * c.x + c.y * c.y);
    }
}

// 归一化内核（除以N）
__global__ void normalize_kernel_td(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] *= scale;
    }
}

// 辅助函数：转换float2_data到float2_td
__host__ void convert_to_cuda_float2(float2_data* src, float2_td* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i].x = src[i].x;
        dst[i].y = src[i].y;
    }
}

// 单对信号互相关函数（原有函数，保持不变）
extern "C" void gpu_correlate_td(float2_data* h_sig1, float2_data* h_sig2, 
                                int input_size, int fft_size,
                                float* h_result, bool normalize) {
    
    // 检查参数
    if (fft_size < input_size) {
        printf("Warning: FFT size (%d) < input size (%d), only first %d samples will be used\n",
               fft_size, input_size, fft_size);
    }
    
    // 确定实际要复制到GPU的数据量
    int copy_size = (input_size > fft_size) ? fft_size : input_size;
    
    // 分配设备内存
    float2_td *d_sig1, *d_sig2;
    size_t copy_bytes = copy_size * sizeof(float2_td);
    size_t fft_bytes = fft_size * sizeof(float2_td);
    
    CUDA_CHECK(cudaMalloc(&d_sig1, fft_bytes));
    CUDA_CHECK(cudaMalloc(&d_sig2, fft_bytes));
    
    // 分配临时主机内存用于转换
    float2_td* h_sig1_cuda = (float2_td*)malloc(copy_bytes);
    float2_td* h_sig2_cuda = (float2_td*)malloc(copy_bytes);
    
    if (!h_sig1_cuda || !h_sig2_cuda) {
        printf("Error: Failed to allocate temporary host memory\n");
        exit(1);
    }
    
    // 转换数据格式
    convert_to_cuda_float2(h_sig1, h_sig1_cuda, copy_size);
    convert_to_cuda_float2(h_sig2, h_sig2_cuda, copy_size);
    
    // 复制输入数据到GPU
    CUDA_CHECK(cudaMemcpy(d_sig1, h_sig1_cuda, copy_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sig2, h_sig2_cuda, copy_bytes, cudaMemcpyHostToDevice));
    
    free(h_sig1_cuda);
    free(h_sig2_cuda);
    
    // 补零
    if (fft_size > copy_size) {
        size_t zero_start_bytes = copy_size * sizeof(float2_td);
        size_t zero_bytes = (fft_size - copy_size) * sizeof(float2_td);
        CUDA_CHECK(cudaMemset((char*)d_sig1 + zero_start_bytes, 0, zero_bytes));
        CUDA_CHECK(cudaMemset((char*)d_sig2 + zero_start_bytes, 0, zero_bytes));
    }
    
    // 创建cuFFT计划
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, fft_size, CUFFT_C2C, 1));
    
    // 前向FFT
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex*)d_sig1, (cufftComplex*)d_sig1, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex*)d_sig2, (cufftComplex*)d_sig2, CUFFT_FORWARD));
    
    // 复数乘法（互功率谱）
    int block_size = 256;
    int grid_size = (fft_size + block_size - 1) / block_size;
    complex_multiply_kernel_td<<<grid_size, block_size>>>((Complex*)d_sig1, (Complex*)d_sig2, fft_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 逆FFT
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex*)d_sig1, (cufftComplex*)d_sig1, CUFFT_INVERSE));
    
    // 取模
    float* d_magnitude;
    CUDA_CHECK(cudaMalloc(&d_magnitude, fft_size * sizeof(float)));
    compute_magnitude_kernel_td<<<grid_size, block_size>>>((Complex*)d_sig1, d_magnitude, fft_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 归一化
    if (normalize) {
        float scale = 1.0f / fft_size;
        normalize_kernel_td<<<grid_size, block_size>>>(d_magnitude, scale, fft_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // 复制结果回CPU
    CUDA_CHECK(cudaMemcpy(h_result, d_magnitude, fft_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 清理
    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_sig1));
    CUDA_CHECK(cudaFree(d_sig2));
    CUDA_CHECK(cudaFree(d_magnitude));
}

// ============ 新增：自相关函数 ============
extern "C" void gpu_auto_correlate_td(float2_data* h_signal, 
                                      int input_size, int fft_size,
                                      float* h_result, bool normalize) {
    
    if (fft_size < input_size) {
        printf("Warning: FFT size (%d) < input size (%d), only first %d samples will be used\n",
               fft_size, input_size, fft_size);
    }
    
    int copy_size = (input_size > fft_size) ? fft_size : input_size;
    
    // 分配设备内存
    float2_td *d_sig;
    size_t copy_bytes = copy_size * sizeof(float2_td);
    size_t fft_bytes = fft_size * sizeof(float2_td);
    
    CUDA_CHECK(cudaMalloc(&d_sig, fft_bytes));
    
    // 转换数据
    float2_td* h_sig_cuda = (float2_td*)malloc(copy_bytes);
    if (!h_sig_cuda) {
        printf("Error: Failed to allocate temporary host memory\n");
        exit(1);
    }
    
    convert_to_cuda_float2(h_signal, h_sig_cuda, copy_size);
    CUDA_CHECK(cudaMemcpy(d_sig, h_sig_cuda, copy_bytes, cudaMemcpyHostToDevice));
    free(h_sig_cuda);
    
    // 补零
    if (fft_size > copy_size) {
        size_t zero_start_bytes = copy_size * sizeof(float2_td);
        size_t zero_bytes = (fft_size - copy_size) * sizeof(float2_td);
        CUDA_CHECK(cudaMemset((char*)d_sig + zero_start_bytes, 0, zero_bytes));
    }
    
    // 创建cuFFT计划
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, fft_size, CUFFT_C2C, 1));
    
    // 前向FFT
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex*)d_sig, (cufftComplex*)d_sig, CUFFT_FORWARD));
    
    // 自相关：乘以自身共轭
    int block_size = 256;
    int grid_size = (fft_size + block_size - 1) / block_size;
    auto_correlation_kernel_td<<<grid_size, block_size>>>((Complex*)d_sig, fft_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 逆FFT
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex*)d_sig, (cufftComplex*)d_sig, CUFFT_INVERSE));
    
    // 取模
    float* d_magnitude;
    CUDA_CHECK(cudaMalloc(&d_magnitude, fft_size * sizeof(float)));
    compute_magnitude_kernel_td<<<grid_size, block_size>>>((Complex*)d_sig, d_magnitude, fft_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 归一化
    if (normalize) {
        float scale = 1.0f / fft_size;
        normalize_kernel_td<<<grid_size, block_size>>>(d_magnitude, scale, fft_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // 复制结果回CPU
    CUDA_CHECK(cudaMemcpy(h_result, d_magnitude, fft_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 清理
    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_sig));
    CUDA_CHECK(cudaFree(d_magnitude));
}

// GPU信息函数（保持不变）
extern "C" void print_gpu_info_td() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    printf("Found %d CUDA device(s):\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        printf("  Device %d: %s\n", i, prop.name);
        printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("    Memory: %.1f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("    Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("    Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        
        if (i == 0) {
            CUDA_CHECK(cudaSetDevice(i));
        }
    }
}

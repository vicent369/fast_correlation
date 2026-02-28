#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <vector>
#include <chrono>
#include <unistd.h>

// 添加头文件包含
#include "fx_correlator.h"

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

// CUDA复数类型
typedef cufftComplex Complex;

// 复数乘法：a * conj(b) (用于互相关)
__global__ void complex_multiply_conj_kernel(
    Complex* a,
    Complex* b,
    Complex* result,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        Complex va = a[idx];
        Complex vb = b[idx];
        
        // result = a * conj(b)
        float real_part = va.x * vb.x + va.y * vb.y;
        float imag_part = va.y * vb.x - va.x * vb.y;
        
        result[idx].x = real_part;
        result[idx].y = imag_part;
    }
}

// 自相关内核（a * conj(a)）
__global__ void auto_correlation_kernel_fx(
    Complex* a,
    Complex* result,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        Complex va = a[idx];
        
        // result = a * conj(a) = |a|^2 (实部)
        result[idx].x = va.x * va.x + va.y * va.y;  // 实部 = |a|^2
        result[idx].y = 0.0f;                         // 虚部 = 0
    }
}

// 复数累加：result += add
__global__ void complex_accumulate_kernel(
    Complex* result,
    Complex* add,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        result[idx].x += add[idx].x;
        result[idx].y += add[idx].y;
    }
}

// 归一化：result *= scale
__global__ void complex_scale_kernel(
    Complex* data,
    float scale,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

// 获取高精度时间（毫秒）
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// ============ 原有函数（保持不变） ============

// FX相关器主函数（互相关）- 非Stream版本
extern "C" void gpu_fx_correlate(
    complex_t* h_sig1,
    complex_t* h_sig2,
    int n_per_frame,
    int num_frames,
    complex_t* h_corr_out,
    int normalize) {
    
    printf("GPU FX Correlator (Cross-correlation): Starting...\n");
    printf("  FFT points per frame: %d\n", n_per_frame);
    printf("  Number of frames: %d\n", num_frames);
    
    // 开始总计时
    double total_start_time = get_time_ms();
    
    // 线程配置
    int block_size = 256;
    int grid_size = (n_per_frame + block_size - 1) / block_size;
    
    // 分配GPU内存
    Complex *d_frame1, *d_frame2;
    Complex *d_fft1, *d_fft2;
    Complex *d_corr_frame;
    Complex *d_corr_accum;
    
    size_t frame_bytes = n_per_frame * sizeof(Complex);
    
    CUDA_CHECK(cudaMalloc(&d_frame1, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_frame2, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_fft1, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_fft2, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_corr_frame, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_corr_accum, frame_bytes));
    
    // 初始化累积数组为0
    CUDA_CHECK(cudaMemset(d_corr_accum, 0, frame_bytes));
    
    // 创建FFT计划
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n_per_frame, CUFFT_C2C, 1));
    
    printf("  Processing frames...\n");
    
    // 统计变量
    double total_transfer_time = 0.0;
    double total_fft_time = 0.0;
    double total_correlation_time = 0.0;
    
    // 处理每一帧
    for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        if (frame_idx % 100 == 0 && frame_idx > 0) {
            double progress = frame_idx * 100.0f / num_frames;
            printf("    Frame %d/%d (%.1f%%)\n", frame_idx, num_frames, progress);
        }
        
        // 计算当前帧在主机内存中的位置
        size_t offset = frame_idx * n_per_frame;
        complex_t* h_frame1 = h_sig1 + offset;
        complex_t* h_frame2 = h_sig2 + offset;
        
        // 数据传输
        double transfer_start = get_time_ms();
        CUDA_CHECK(cudaMemcpy(d_frame1, h_frame1, frame_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_frame2, h_frame2, frame_bytes, cudaMemcpyHostToDevice));
        double transfer_end = get_time_ms();
        total_transfer_time += (transfer_end - transfer_start);
        
        // FFT
        double fft_start = get_time_ms();
        CUFFT_CHECK(cufftExecC2C(plan, d_frame1, d_fft1, CUFFT_FORWARD));
        CUFFT_CHECK(cufftExecC2C(plan, d_frame2, d_fft2, CUFFT_FORWARD));
        double fft_end = get_time_ms();
        total_fft_time += (fft_end - fft_start);
        
        // 互相关计算
        double corr_start = get_time_ms();
        complex_multiply_conj_kernel<<<grid_size, block_size>>>(
            d_fft1, d_fft2, d_corr_frame, n_per_frame);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 累加
        complex_accumulate_kernel<<<grid_size, block_size>>>(
            d_corr_accum, d_corr_frame, n_per_frame);
        CUDA_CHECK(cudaDeviceSynchronize());
        double corr_end = get_time_ms();
        total_correlation_time += (corr_end - corr_start);
    }
    
    // 归一化
    if (normalize) {
        float scale = 1.0f / n_per_frame;
        complex_scale_kernel<<<grid_size, block_size>>>(
            d_corr_accum, scale, n_per_frame);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // 结果传回CPU
    double output_start = get_time_ms();
    CUDA_CHECK(cudaMemcpy(h_corr_out, d_corr_accum, frame_bytes, cudaMemcpyDeviceToHost));
    double output_end = get_time_ms();
    
    // 结束总计时
    double total_end_time = get_time_ms();
    double total_elapsed_ms = total_end_time - total_start_time;
    
    // 清理
    CUFFT_CHECK(cufftDestroy(plan));
    
    CUDA_CHECK(cudaFree(d_frame1));
    CUDA_CHECK(cudaFree(d_frame2));
    CUDA_CHECK(cudaFree(d_fft1));
    CUDA_CHECK(cudaFree(d_fft2));
    CUDA_CHECK(cudaFree(d_corr_frame));
    CUDA_CHECK(cudaFree(d_corr_accum));
    
    printf("GPU FX Correlator: Completed in %.2f ms\n", total_elapsed_ms);
}

// 自相关函数 - 非Stream版本
extern "C" void gpu_fx_auto_correlate(
    complex_t* h_signal,
    int n_per_frame,
    int num_frames,
    complex_t* h_auto_out,
    int normalize) {
    
    printf("GPU FX Auto-correlator: Starting...\n");
    printf("  FFT points per frame: %d\n", n_per_frame);
    printf("  Number of frames: %d\n", num_frames);
    
    // 线程配置
    int block_size = 256;
    int grid_size = (n_per_frame + block_size - 1) / block_size;
    
    // 分配GPU内存
    Complex *d_frame;
    Complex *d_fft;
    Complex *d_auto_frame;
    Complex *d_auto_accum;
    
    size_t frame_bytes = n_per_frame * sizeof(Complex);
    
    CUDA_CHECK(cudaMalloc(&d_frame, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_fft, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_auto_frame, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_auto_accum, frame_bytes));
    
    // 初始化累积数组为0
    CUDA_CHECK(cudaMemset(d_auto_accum, 0, frame_bytes));
    
    // 创建FFT计划
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n_per_frame, CUFFT_C2C, 1));
    
    // 处理每一帧
    for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        size_t offset = frame_idx * n_per_frame;
        complex_t* h_frame = h_signal + offset;
        
        // 数据传输
        CUDA_CHECK(cudaMemcpy(d_frame, h_frame, frame_bytes, cudaMemcpyHostToDevice));
        
        // FFT
        CUFFT_CHECK(cufftExecC2C(plan, d_frame, d_fft, CUFFT_FORWARD));
        
        // 自相关：乘以自身共轭
        auto_correlation_kernel_fx<<<grid_size, block_size>>>(
            d_fft, d_auto_frame, n_per_frame);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 累加
        complex_accumulate_kernel<<<grid_size, block_size>>>(
            d_auto_accum, d_auto_frame, n_per_frame);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // 归一化
    if (normalize) {
        float scale = 1.0f / n_per_frame;
        complex_scale_kernel<<<grid_size, block_size>>>(
            d_auto_accum, scale, n_per_frame);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // 结果传回CPU
    CUDA_CHECK(cudaMemcpy(h_auto_out, d_auto_accum, frame_bytes, cudaMemcpyDeviceToHost));
    
    // 清理
    CUFFT_CHECK(cufftDestroy(plan));
    
    CUDA_CHECK(cudaFree(d_frame));
    CUDA_CHECK(cudaFree(d_fft));
    CUDA_CHECK(cudaFree(d_auto_frame));
    CUDA_CHECK(cudaFree(d_auto_accum));
    
    printf("GPU FX Auto-correlator: Completed\n");
}

// ============ 新增：Stream版本函数 ============

// Stream版本互相关
extern "C" void gpu_fx_correlate_stream(
    complex_t* h_sig1,
    complex_t* h_sig2,
    int n_per_frame,
    int num_frames,
    complex_t* h_corr_out,
    int normalize,
    cudaStream_t stream
) {
    size_t frame_bytes = n_per_frame * sizeof(Complex);
    
    // 线程配置
    int block_size = 256;
    int grid_size = (n_per_frame + block_size - 1) / block_size;
    
    // 为这个流分配资源
    Complex *d_frame1, *d_frame2;
    Complex *d_fft1, *d_fft2;
    Complex *d_corr_frame;
    Complex *d_corr_accum;
    
    CUDA_CHECK(cudaMallocAsync(&d_frame1, frame_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_frame2, frame_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_fft1, frame_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_fft2, frame_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_corr_frame, frame_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_corr_accum, frame_bytes, stream));
    
    // 初始化累积数组为0
    CUDA_CHECK(cudaMemsetAsync(d_corr_accum, 0, frame_bytes, stream));
    
    // 创建FFT计划并绑定到流
    cufftHandle plan;
    cufftCreate(&plan);
    cufftSetStream(plan, stream);
    cufftPlan1d(&plan, n_per_frame, CUFFT_C2C, 1);
    
    // 处理每一帧
    for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        size_t offset = frame_idx * n_per_frame;
        
        // 异步传输当前帧
        CUDA_CHECK(cudaMemcpyAsync(d_frame1, h_sig1 + offset, 
                                   frame_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_frame2, h_sig2 + offset, 
                                   frame_bytes, cudaMemcpyHostToDevice, stream));
        
        // FFT（在同一个流中）
        CUFFT_CHECK(cufftExecC2C(plan, d_frame1, d_fft1, CUFFT_FORWARD));
        CUFFT_CHECK(cufftExecC2C(plan, d_frame2, d_fft2, CUFFT_FORWARD));
        
        // 互相关计算
        complex_multiply_conj_kernel<<<grid_size, block_size, 0, stream>>>(
            d_fft1, d_fft2, d_corr_frame, n_per_frame);
        
        // 累加
        complex_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
            d_corr_accum, d_corr_frame, n_per_frame);
    }
    
    // 归一化
    if (normalize) {
        float scale = 1.0f / n_per_frame;
        complex_scale_kernel<<<grid_size, block_size, 0, stream>>>(
            d_corr_accum, scale, n_per_frame);
    }
    
    // 结果传回CPU
    CUDA_CHECK(cudaMemcpyAsync(h_corr_out, d_corr_accum, 
                               frame_bytes, cudaMemcpyDeviceToHost, stream));
    
    // 等待这个流完成
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // 清理
    cufftDestroy(plan);
    CUDA_CHECK(cudaFreeAsync(d_frame1, stream));
    CUDA_CHECK(cudaFreeAsync(d_frame2, stream));
    CUDA_CHECK(cudaFreeAsync(d_fft1, stream));
    CUDA_CHECK(cudaFreeAsync(d_fft2, stream));
    CUDA_CHECK(cudaFreeAsync(d_corr_frame, stream));
    CUDA_CHECK(cudaFreeAsync(d_corr_accum, stream));
}

// Stream版本自相关
extern "C" void gpu_fx_auto_correlate_stream(
    complex_t* h_signal,
    int n_per_frame,
    int num_frames,
    complex_t* h_auto_out,
    int normalize,
    cudaStream_t stream
) {
    size_t frame_bytes = n_per_frame * sizeof(Complex);
    
    int block_size = 256;
    int grid_size = (n_per_frame + block_size - 1) / block_size;
    
    Complex *d_frame, *d_fft, *d_auto_frame, *d_auto_accum;
    
    CUDA_CHECK(cudaMallocAsync(&d_frame, frame_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_fft, frame_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_auto_frame, frame_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_auto_accum, frame_bytes, stream));
    
    CUDA_CHECK(cudaMemsetAsync(d_auto_accum, 0, frame_bytes, stream));
    
    cufftHandle plan;
    cufftCreate(&plan);
    cufftSetStream(plan, stream);
    cufftPlan1d(&plan, n_per_frame, CUFFT_C2C, 1);
    
    for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        size_t offset = frame_idx * n_per_frame;
        
        CUDA_CHECK(cudaMemcpyAsync(d_frame, h_signal + offset, 
                                   frame_bytes, cudaMemcpyHostToDevice, stream));
        
        CUFFT_CHECK(cufftExecC2C(plan, d_frame, d_fft, CUFFT_FORWARD));
        
        auto_correlation_kernel_fx<<<grid_size, block_size, 0, stream>>>(
            d_fft, d_auto_frame, n_per_frame);
        
        complex_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
            d_auto_accum, d_auto_frame, n_per_frame);
    }
    
    if (normalize) {
        float scale = 1.0f / n_per_frame;
        complex_scale_kernel<<<grid_size, block_size, 0, stream>>>(
            d_auto_accum, scale, n_per_frame);
    }
    
    CUDA_CHECK(cudaMemcpyAsync(h_auto_out, d_auto_accum, 
                               frame_bytes, cudaMemcpyDeviceToHost, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    cufftDestroy(plan);
    CUDA_CHECK(cudaFreeAsync(d_frame, stream));
    CUDA_CHECK(cudaFreeAsync(d_fft, stream));
    CUDA_CHECK(cudaFreeAsync(d_auto_frame, stream));
    CUDA_CHECK(cudaFreeAsync(d_auto_accum, stream));
}

// 批量处理所有相关对（使用多个流并行）
extern "C" void gpu_fx_correlate_batch(
    std::vector<complex_t*>& h_channel_data,
    std::vector<FxCorrelationPairResult>& all_pairs,
    int n_per_frame,
    int num_frames,
    int normalize
) {
    int num_pairs = all_pairs.size();
    int max_streams = std::min(num_pairs, 4);  // GTX 1080 Ti 建议最多4个流
    
    printf("  Using %d CUDA streams for parallel processing\n", max_streams);
    
    // 创建流
    std::vector<cudaStream_t> streams(max_streams);
    for (int i = 0; i < max_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // 记录每个流当前处理的对索引
    std::vector<int> stream_task(max_streams, -1);
    std::vector<bool> stream_busy(max_streams, false);
    
    int completed = 0;
    int next_pair = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (completed < num_pairs) {
        // 分配新任务到空闲流
        for (int s = 0; s < max_streams; s++) {
            if (!stream_busy[s] && next_pair < num_pairs) {
                int pair_idx = next_pair++;
                auto& pair = all_pairs[pair_idx];
                
                // 在指定流中启动任务
                if (pair.type == FX_AUTO_CORRELATION) {
                    gpu_fx_auto_correlate_stream(
                        h_channel_data[pair.channel_i],
                        n_per_frame, num_frames,
                        pair.accumulated_spectrum.data(),
                        normalize,
                        streams[s]
                    );
                } else {
                    gpu_fx_correlate_stream(
                        h_channel_data[pair.channel_i],
                        h_channel_data[pair.channel_j],
                        n_per_frame, num_frames,
                        pair.accumulated_spectrum.data(),
                        normalize,
                        streams[s]
                    );
                }
                
                stream_task[s] = pair_idx;
                stream_busy[s] = true;
            }
        }
        
        // 检查完成的流
        for (int s = 0; s < max_streams; s++) {
            if (stream_busy[s]) {
                cudaError_t err = cudaStreamQuery(streams[s]);
                if (err != cudaErrorNotReady) {
                    // 流已完成
                    stream_busy[s] = false;
                    completed++;
                    
                    // 计算该对的统计信息
                    auto& pair = all_pairs[stream_task[s]];
                    float total_power = 0;
                    float max_mag = 0;
                    int max_idx = 0;
                    
                    for (int i = 0; i < n_per_frame; i++) {
                        float real = pair.accumulated_spectrum[i].x;
                        float imag = pair.accumulated_spectrum[i].y;
                        float mag = sqrtf(real*real + imag*imag);
                        total_power += mag;
                        if (mag > max_mag) {
                            max_mag = mag;
                            max_idx = i;
                        }
                    }
                }
            }
        }
        
        // 避免CPU空转
        usleep(1000);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(end_time - start_time);
    
    printf("  Batch processing completed in %.2f ms\n", duration.count());
    
    // 清理流
    for (int i = 0; i < max_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

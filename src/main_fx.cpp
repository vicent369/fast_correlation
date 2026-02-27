#include "dat_reader.h"
#include "fx_correlator.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <ctime>
#include <sys/stat.h>
#include <cstring>

// 删除或注释掉这行，因为已经在 fx_correlator.h 中定义了
// typedef float2_data complex_t;

// 保存复数数组到CSV
void save_complex_spectrum_csv(const complex_t* data, int n, 
                              float sample_rate_hz,
                              const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return;
    }
    
    fprintf(fp, "frequency_index,frequency_hz,real_part,imag_part,magnitude,phase_deg\n");
    
    for (int i = 0; i < n; i++) {
        float real = data[i].x;  // 使用 .x 访问实部
        float imag = data[i].y;  // 使用 .y 访问虚部
        float magnitude = sqrtf(real*real + imag*imag);
        float phase_deg = atan2f(imag, real) * 180.0f / M_PI;
        
        // 频率计算（考虑到复数FFT的对称性）
        float frequency_hz;
        if (i <= n/2) {
            frequency_hz = i * sample_rate_hz / n;
        } else {
            frequency_hz = (i - n) * sample_rate_hz / n;
        }
        
        fprintf(fp, "%d,%.1f,%.6e,%.6e,%.6e,%.2f\n",
                i, frequency_hz, real, imag, magnitude, phase_deg);
    }
    
    fclose(fp);
    printf("  ✓ Saved complex spectrum to: %s\n", filename.c_str());
}

// 保存二进制文件
void save_complex_spectrum_binary(const complex_t* data, int n,
                                 const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return;
    }
    
    fwrite(data, sizeof(complex_t), n, fp);
    fclose(fp);
    
    printf("  ✓ Saved binary complex data to: %s (%d points, %.1f MB)\n",
           filename.c_str(), n, n * sizeof(complex_t) / 1024.0 / 1024.0);
}

// 创建输出目录
void create_output_dir(const std::string& dirname) {
    struct stat st = {0};
    if (stat(dirname.c_str(), &st) == -1) {
        mkdir(dirname.c_str(), 0755);
    }
}

// 提取基本文件名
std::string get_base_filename(const std::string& fullpath) {
    std::string filename = fullpath;
    
    // 去掉路径
    size_t slash_pos = filename.find_last_of("/\\");
    if (slash_pos != std::string::npos) {
        filename = filename.substr(slash_pos + 1);
    }
    
    // 去掉扩展名
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos != std::string::npos) {
        filename = filename.substr(0, dot_pos);
    }
    
    return filename;
}

// 辅助函数：打印分隔线
void print_separator(int length, char ch) {
    for (int i = 0; i < length; i++) {
        printf("%c", ch);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    print_separator(70, '=');
    printf("                 FX CORRELATOR (Frequency Domain)\n");
    printf("           Implements Algorithm 2: calcCorr = Y1 × conj(Y2)\n");
    print_separator(70, '=');
    printf("\n");
    
    // 参数解析
    if (argc < 3) {
        printf("Usage: %s <file1> <file2> [n_per_frame] [num_frames] [start_sample]\n", argv[0]);
        printf("\nParameters:\n");
        printf("  file1, file2      : Input DAT files (IQ int16 format)\n");
        printf("  n_per_frame       : FFT points per frame (default: 1024)\n");
        printf("  num_frames        : Number of frames to accumulate (default: 1000)\n");
        printf("  start_sample      : Starting sample offset (default: 0)\n");
        printf("\nExample:\n");
        printf("  %s a0.dat b0.dat 1024 1000 0\n", argv[0]);
        printf("  %s a0.dat b0.dat 4096 500      # 4096-point FFT, 500 frames\n", argv[0]);
        return 1;
    }
    
    const char* file1 = argv[1];
    const char* file2 = argv[2];
    
    // 默认参数
    int n_per_frame = 1024;      // 每帧FFT点数
    int num_frames = 1000;       // 累积帧数
    size_t start_sample = 0;     // 起始样本
    int normalize = 1;           // 是否归一化
    
    if (argc > 3) n_per_frame = atoi(argv[3]);
    if (argc > 4) num_frames = atoi(argv[4]);
    if (argc > 5) start_sample = atol(argv[5]);
    
    printf("Input Parameters:\n");
    printf("  File 1: %s\n", file1);
    printf("  File 2: %s\n", file2);
    printf("  FFT points per frame (N_freq): %d\n", n_per_frame);
    printf("  Number of frames to accumulate: %d\n", num_frames);
    printf("  Starting sample offset: %zu\n", start_sample);
    printf("  Normalize FFT: %s\n", normalize ? "Yes" : "No");
    
    // 计算总样本数
    size_t total_samples = n_per_frame * num_frames;
    printf("\nMemory Requirements:\n");
    printf("  Total samples per file: %zu (%.1f MB)\n", 
           total_samples, total_samples * sizeof(complex_t) / 1024.0 / 1024.0);
    printf("  Total input data: %.1f MB\n", 
           total_samples * sizeof(complex_t) * 2 / 1024.0 / 1024.0);
    
    // 创建输出目录
    create_output_dir("fx_correlation_results");
    
    // 生成时间戳和文件名
    time_t raw_time;
    time(&raw_time);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&raw_time));
    
    std::string base1 = get_base_filename(file1);
    std::string base2 = get_base_filename(file2);
    std::string prefix = "fx_correlation_results/" + base1 + "_vs_" + base2 + "_" + timestamp;
    
    // 分配内存
    printf("\nAllocating memory...\n");
    // 注意：需要强制转换为 complex_t*，因为 malloc 返回的是 void*
    complex_t* sig1_data = (complex_t*)malloc(total_samples * sizeof(complex_t));
    complex_t* sig2_data = (complex_t*)malloc(total_samples * sizeof(complex_t));
    complex_t* corr_result = (complex_t*)malloc(n_per_frame * sizeof(complex_t));
    
    if (!sig1_data || !sig2_data || !corr_result) {
        printf("Error: Failed to allocate memory\n");
        return 1;
    }
    
    // 读取数据
    printf("Reading data from files...\n");
    
    size_t samples_read = 0;
    for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        size_t frame_start = start_sample + frame_idx * n_per_frame;
        
        // 读取当前帧 - 使用 float2_data 类型读取
        float2_data* frame1 = read_dat_partial(file1, frame_start, n_per_frame);
        float2_data* frame2 = read_dat_partial(file2, frame_start, n_per_frame);
        
        if (!frame1 || !frame2) {
            printf("Error: Failed to read frame %d\n", frame_idx);
            free(sig1_data);
            free(sig2_data);
            free(corr_result);
            return 1;
        }
        
        // 复制到总数组 - 需要类型转换
        size_t offset = frame_idx * n_per_frame;
        // 将 float2_data 转换为 complex_t
        for (int i = 0; i < n_per_frame; i++) {
            sig1_data[offset + i].x = frame1[i].x;
            sig1_data[offset + i].y = frame1[i].y;
            
            sig2_data[offset + i].x = frame2[i].x;
            sig2_data[offset + i].y = frame2[i].y;
        }
        
        free_data(frame1);
        free_data(frame2);
        
        samples_read += n_per_frame;
        
        if (frame_idx % 100 == 0) {
            printf("  Read %zu samples (%.1f%%)\n", 
                   samples_read, samples_read * 100.0 / total_samples);
        }
    }
    
    printf("  Total samples read: %zu (%.1f MB)\n", 
           samples_read, samples_read * sizeof(complex_t) / 1024.0 / 1024.0);
    
    // GPU计算
    printf("\nComputing FX correlation on GPU...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // 调用 GPU 函数，它会显示详细的性能信息
    gpu_fx_correlate(sig1_data, sig2_data, n_per_frame, num_frames, corr_result, normalize);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(end - start);

    // 获取总体统计（包括 Python 调用开销）
    float total_ms = duration.count();
    float total_seconds = total_ms / 1000.0;
    // 使用不同的变量名避免冲突
    long long processed_samples_total = (long long)n_per_frame * (long long)num_frames;
    float total_throughput = processed_samples_total / total_seconds;

    printf("\nOverall Performance (including Python overhead):\n");
    printf("  Total elapsed time: %.2f ms (%.3f seconds)\n", total_ms, total_seconds);
    printf("  Samples processed: %lld\n", processed_samples_total);
    printf("  Overall throughput: %.2f MS/s\n", total_throughput / 1e6);
    printf("  Average frames per second: %.1f\n", num_frames / total_seconds);
    
    // 计算统计信息
    printf("\nComputing statistics...\n");
    float total_power = 0;
    float max_magnitude = 0;
    int max_index = 0;
    
    for (int i = 0; i < n_per_frame; i++) {
        float real = corr_result[i].x;
        float imag = corr_result[i].y;
        float magnitude = sqrtf(real*real + imag*imag);
        
        total_power += magnitude;
        if (magnitude > max_magnitude) {
            max_magnitude = magnitude;
            max_index = i;
        }
    }
    
    float sample_rate_hz = 100e6;  // 假设100MHz采样率
    float freq_resolution = sample_rate_hz / n_per_frame;
    
    // 最大幅度对应的频率
    float max_freq_hz;
    if (max_index <= n_per_frame/2) {
        max_freq_hz = max_index * freq_resolution;
    } else {
        max_freq_hz = (max_index - n_per_frame) * freq_resolution;
    }
    
    printf("\nCorrelation Spectrum Statistics:\n");
    printf("  Total integrated power: %.6e\n", total_power);
    printf("  Maximum magnitude: %.6e at index %d\n", max_magnitude, max_index);
    printf("  Frequency at max: %.1f Hz (%.3f MHz)\n", max_freq_hz, max_freq_hz/1e6);
    printf("  Average magnitude: %.6e\n", total_power / n_per_frame);
    printf("  Frequency resolution: %.1f Hz (%.3f kHz)\n", 
           freq_resolution, freq_resolution/1e3);
    
    // 保存结果
    printf("\nSaving results...\n");
    
    // 1. 复数谱CSV
    std::string csv_filename = prefix + "_complex_spectrum.csv";
    save_complex_spectrum_csv(corr_result, n_per_frame, sample_rate_hz, csv_filename);
    
    // 2. 二进制文件
    std::string bin_filename = prefix + "_complex_spectrum.bin";
    save_complex_spectrum_binary(corr_result, n_per_frame, bin_filename);
    
    // 3. 统计摘要
    std::string txt_filename = prefix + "_summary.txt";
    FILE* summary_fp = fopen(txt_filename.c_str(), "w");
    if (summary_fp) {
        fprintf(summary_fp, "FX CORRELATOR RESULTS SUMMARY\n");
        fprintf(summary_fp, "==============================\n\n");
        fprintf(summary_fp, "Input Files:\n");
        fprintf(summary_fp, "  %s\n", file1);
        fprintf(summary_fp, "  %s\n\n", file2);
        
        fprintf(summary_fp, "Processing Parameters:\n");
        fprintf(summary_fp, "  FFT points per frame: %d\n", n_per_frame);
        fprintf(summary_fp, "  Number of frames: %d\n", num_frames);
        fprintf(summary_fp, "  Total samples: %zu\n", total_samples);
        fprintf(summary_fp, "  Start sample: %zu\n", start_sample);
        fprintf(summary_fp, "  Normalize FFT: %s\n\n", normalize ? "Yes" : "No");
        
        fprintf(summary_fp, "Performance:\n");
        fprintf(summary_fp, "  Processing time: %.2f ms\n", duration.count());
        fprintf(summary_fp, "  Samples per second: %.2f MS/s\n", 
                total_samples / (duration.count() / 1000.0) / 1e6);
        fprintf(summary_fp, "  Frames per second: %.1f\n", 
                num_frames / (duration.count() / 1000.0));
        fprintf(summary_fp, "\n");
        
        fprintf(summary_fp, "Spectrum Statistics:\n");
        fprintf(summary_fp, "  Total integrated power: %.6e\n", total_power);
        fprintf(summary_fp, "  Maximum magnitude: %.6e\n", max_magnitude);
        fprintf(summary_fp, "  Maximum at index: %d\n", max_index);
        fprintf(summary_fp, "  Frequency at maximum: %.1f Hz (%.3f MHz)\n", 
                max_freq_hz, max_freq_hz/1e6);
        fprintf(summary_fp, "  Average magnitude: %.6e\n", total_power / n_per_frame);
        fprintf(summary_fp, "  Frequency resolution: %.1f Hz\n\n", freq_resolution);
        
        fprintf(summary_fp, "Output Files:\n");
        fprintf(summary_fp, "  %s - Complex spectrum (CSV)\n", csv_filename.c_str());
        fprintf(summary_fp, "  %s - Complex spectrum (binary)\n", bin_filename.c_str());
        fprintf(summary_fp, "  %s - This summary\n\n", txt_filename.c_str());
        
        fprintf(summary_fp, "Timestamp: %s\n", timestamp);
        
        fclose(summary_fp);
        printf("  ✓ Saved summary to: %s\n", txt_filename.c_str());
    }
    
    // 清理
    printf("\nCleaning up...\n");
    free(sig1_data);
    free(sig2_data);
    free(corr_result);
    
    printf("\n");
    print_separator(60, '=');
    printf("✅ FX CORRELATOR COMPLETED SUCCESSFULLY\n");
    printf("   Results saved in: fx_correlation_results/\n");
    print_separator(60, '=');
    printf("\n");
    
    return 0;
}

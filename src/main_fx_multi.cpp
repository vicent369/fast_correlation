#include "dat_reader.h"
#include "fx_correlator.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>

// 函数声明
void create_fx_output_directory(const std::string& dirname);
std::string get_fx_base_filename(const std::string& fullpath);
bool save_fx_complex_spectrum_csv(const complex_t* data, int n, 
                                  float sample_rate_hz,
                                  const std::string& filename,
                                  const std::string& pair_name);
bool save_fx_complex_spectrum_binary(const complex_t* data, int n,
                                     const std::string& filename,
                                     const std::string& pair_name);
bool save_fx_multi_channel_config(const FxMultiChannelConfig& config,
                                  const std::string& filename);
bool save_fx_multi_channel_summary_txt(const FxMultiChannelStats& stats,
                                       const std::string& filename);

// 辅助函数
void print_line(int length = 60, char ch = '=') {
    std::cout << std::string(length, ch) << std::endl;
}

void print_double_line(int length = 60, char ch = '=') {
    std::cout << "\n" << std::string(length, ch) << std::endl;
}

// 计算复数谱的统计信息
void compute_spectrum_stats(const complex_t* spectrum, int n, 
                           float sample_rate_hz,
                           float& total_power, float& max_magnitude,
                           int& max_index, float& max_freq_hz) {
    total_power = 0.0f;
    max_magnitude = 0.0f;
    max_index = 0;
    
    for (int i = 0; i < n; i++) {
        float real = spectrum[i].x;
        float imag = spectrum[i].y;
        float magnitude = sqrtf(real*real + imag*imag);
        
        total_power += magnitude;
        if (magnitude > max_magnitude) {
            max_magnitude = magnitude;
            max_index = i;
        }
    }
    
    float freq_resolution = sample_rate_hz / n;
    if (max_index <= n/2) {
        max_freq_hz = max_index * freq_resolution;
    } else {
        max_freq_hz = (max_index - n) * freq_resolution;
    }
}

// ============ 新增：一次性读取整个文件 ============
float2_data* read_entire_file(const char* filename, size_t* num_samples) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return nullptr;
    }
    
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    rewind(fp);
    
    *num_samples = file_size / 4;  // 每个样本4字节
    size_t total_samples = *num_samples;
    
    printf("    File size: %.2f MB, Samples: %zu\n", 
           file_size / 1024.0 / 1024.0, total_samples);
    
    // 一次性读取所有原始数据
    short* raw_data = (short*)malloc(file_size);
    if (!raw_data) {
        printf("Error: Failed to allocate memory for raw data\n");
        fclose(fp);
        return nullptr;
    }
    
    size_t read_size = fread(raw_data, 1, file_size, fp);
    fclose(fp);
    
    if (read_size != file_size) {
        printf("Error: Failed to read entire file\n");
        free(raw_data);
        return nullptr;
    }
    
    // 转换为复数
    float2_data* complex_data = (float2_data*)malloc(total_samples * sizeof(float2_data));
    if (!complex_data) {
        printf("Error: Failed to allocate memory for complex data\n");
        free(raw_data);
        return nullptr;
    }
    
    // 批量转换
    for (size_t i = 0; i < total_samples; i++) {
        complex_data[i].x = (float)raw_data[2*i];
        complex_data[i].y = (float)raw_data[2*i + 1];
    }
    
    free(raw_data);
    return complex_data;
}

// 多通道频域处理函数（优化版 - 不保存原始数据）
bool process_fx_multi_channel(const FxMultiChannelConfig& config,
                             int n_per_frame, int num_frames,
                             int normalize, size_t start_sample,
                             float sample_rate_hz = 100e6) {
    
    print_double_line(70, '=');
    std::cout << "FX MULTI-CHANNEL PROCESSING (Frequency Domain) - OPTIMIZED" << std::endl;
    print_line(70, '=');
    
    printf("Number of channels: %d\n", config.num_channels);
    printf("Total pairs: %d (Auto: %d, Cross: %d)\n", 
           config.get_total_pairs(), config.num_channels,
           config.num_channels * (config.num_channels - 1) / 2);
    printf("FFT points per frame: %d\n", n_per_frame);
    printf("Number of frames: %d\n", num_frames);
    printf("Total samples needed: %d\n", n_per_frame * num_frames);
    printf("Start sample offset: %zu\n\n", start_sample);
    
    // 计算需要的样本数
    size_t samples_needed = n_per_frame * num_frames;
    
    // 创建输出目录
    create_fx_output_directory("fx_multi_results");
    
    // 生成时间戳
    time_t raw_time;
    time(&raw_time);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&raw_time));
    
    std::string prefix = "fx_multi_results/fx_multi_" + std::string(timestamp);
    
    // 保存配置
    save_fx_multi_channel_config(config, prefix + "_config.csv");
    
    // 分配处理用的内存
    printf("\nAllocating processing memory...\n");
    std::vector<complex_t*> channel_data(config.num_channels, nullptr);
    for (int c = 0; c < config.num_channels; c++) {
        channel_data[c] = (complex_t*)malloc(samples_needed * sizeof(complex_t));
        if (!channel_data[c]) {
            printf("Error: Failed to allocate memory for channel %d\n", c);
            return false;
        }
    }
    
    // 读取所有通道的数据
    printf("\nReading data from files...\n");
    auto read_start = std::chrono::high_resolution_clock::now();
    
    for (int c = 0; c < config.num_channels; c++) {
        printf("  Channel %d (%s):\n", c+1, config.channel_names[c].c_str());
        
        // 打开文件
        FILE* fp = fopen(config.channel_files[c].c_str(), "rb");
        if (!fp) {
            printf("Error: Cannot open file %s\n", config.channel_files[c].c_str());
            return false;
        }
        
        // 定位到起始位置
        size_t start_byte = start_sample * 4;  // 每个样本4字节
        if (fseek(fp, start_byte, SEEK_SET) != 0) {
            printf("Error: Failed to seek in file\n");
            fclose(fp);
            return false;
        }
        
        // 读取所有需要的样本
        size_t bytes_to_read = samples_needed * 4;
        short* raw_buffer = (short*)malloc(bytes_to_read);
        if (!raw_buffer) {
            printf("Error: Failed to allocate raw buffer\n");
            fclose(fp);
            return false;
        }
        
        size_t bytes_read = fread(raw_buffer, 1, bytes_to_read, fp);
        fclose(fp);
        
        if (bytes_read != bytes_to_read) {
            printf("Warning: Read %zu of %zu bytes\n", bytes_read, bytes_to_read);
            samples_needed = bytes_read / 4;
            printf("  Adjusting to %zu samples\n", samples_needed);
        }
        
        // 转换为complex_t格式
        size_t samples_actual = bytes_read / 4;
        for (size_t i = 0; i < samples_actual; i++) {
            channel_data[c][i].x = (float)raw_buffer[2*i];
            channel_data[c][i].y = (float)raw_buffer[2*i + 1];
        }
        
        free(raw_buffer);
        printf("    Loaded %zu samples\n", samples_actual);
    }
    
    auto read_end = std::chrono::high_resolution_clock::now();
    auto read_ms = std::chrono::duration<float, std::milli>(read_end - read_start).count();
    
    printf("Data loading time: %.2f ms (%.2f MB/s)\n", 
           read_ms, (samples_needed * config.num_channels * 8) / read_ms / 1000);
    
    // 更新总样本数和帧数
    size_t total_samples = samples_needed;
    int actual_frames = total_samples / n_per_frame;
    if (actual_frames != num_frames) {
        printf("\nNote: Adjusted frames from %d to %d\n", num_frames, actual_frames);
        num_frames = actual_frames;
    }
    
    // 创建所有相关对的结果容器
    std::vector<FxCorrelationPairResult> all_pairs;
    
    // 添加自相关对
    for (int i = 0; i < config.num_channels; i++) {
        FxCorrelationPairResult pair;
        pair.type = FX_AUTO_CORRELATION;
        pair.channel_i = i;
        pair.channel_j = i;
        pair.pair_name = config.channel_names[i] + "_AUTO";
        pair.spectrum_size = n_per_frame;
        pair.accumulated_spectrum.resize(n_per_frame);
        all_pairs.push_back(pair);
    }
    
    // 添加互相关对
    for (int i = 0; i < config.num_channels; i++) {
        for (int j = i + 1; j < config.num_channels; j++) {
            FxCorrelationPairResult pair;
            pair.type = FX_CROSS_CORRELATION;
            pair.channel_i = i;
            pair.channel_j = j;
            pair.pair_name = config.channel_names[i] + "x" + config.channel_names[j];
            pair.spectrum_size = n_per_frame;
            pair.accumulated_spectrum.resize(n_per_frame);
            all_pairs.push_back(pair);
        }
    }
    
    // GPU计算 - 使用批量处理
    printf("\nComputing FX correlations on GPU...\n");
    gpu_fx_correlate_batch(channel_data, all_pairs, n_per_frame, num_frames, normalize);
    
    // 计算统计信息并保存结果
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (auto& pair : all_pairs) {
        // 计算统计信息
        float total_power = 0.0f;
        float max_magnitude = 0.0f;
        int max_index = 0;
        
        for (int i = 0; i < n_per_frame; i++) {
            float real = pair.accumulated_spectrum[i].x;
            float imag = pair.accumulated_spectrum[i].y;
            float mag = sqrtf(real*real + imag*imag);
            total_power += mag;
            if (mag > max_magnitude) {
                max_magnitude = mag;
                max_index = i;
            }
        }
        
        float freq_resolution = sample_rate_hz / n_per_frame;
        float max_freq_hz;
        if (max_index <= n_per_frame/2) {
            max_freq_hz = max_index * freq_resolution;
        } else {
            max_freq_hz = (max_index - n_per_frame) * freq_resolution;
        }
        
        pair.total_power = total_power;
        pair.max_magnitude = max_magnitude;
        pair.max_index = max_index;
        pair.max_freq_hz = max_freq_hz;
        
        // 保存结果
        std::string bin_filename = prefix + "_" + pair.pair_name + "_spectrum.bin";
        save_fx_complex_spectrum_binary(pair.accumulated_spectrum.data(),
                                       n_per_frame, bin_filename, pair.pair_name);
        
        std::string csv_filename = prefix + "_" + pair.pair_name + "_spectrum.csv";
        save_fx_complex_spectrum_csv(pair.accumulated_spectrum.data(),
                                    n_per_frame, sample_rate_hz,
                                    csv_filename, pair.pair_name);
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration<float, std::milli>(total_end - total_start);
    
    // 保存统计摘要
    FxMultiChannelStats stats;
    stats.config = config;
    stats.pairs = all_pairs;
    stats.n_per_frame = n_per_frame;
    stats.num_frames = num_frames;
    stats.total_samples = total_samples;
    stats.processing_time_ms = total_duration.count();
    stats.processing_rate_msps = total_samples * config.num_channels / 
                                (total_duration.count() / 1000.0) / 1e6;
    stats.sample_rate_hz = sample_rate_hz;
    stats.timestamp_str = timestamp;
    
    save_fx_multi_channel_summary_txt(stats, prefix + "_summary.txt");
    
    // 显示最终统计
    print_double_line(60, '=');
    std::cout << "FX MULTI-CHANNEL PROCESSING COMPLETE" << std::endl;
    print_line(60, '=');
    
    printf("Data loading: %.2f ms\n", read_ms);
    printf("GPU processing: %.2f ms\n", total_duration.count());
    printf("Total time: %.2f ms\n", read_ms + total_duration.count());
    printf("Processing rate: %.2f MS/s\n", stats.processing_rate_msps);
    printf("\nResults saved to: fx_multi_results/\n");
    printf("Note: Raw data NOT saved (disabled for performance)\n");
    
    // 显示各对结果
    printf("\nSpectrum Statistics:\n");
    for (const auto& pair : all_pairs) {
        printf("  %s:\n", pair.pair_name.c_str());
        printf("    Max magnitude: %.3e at %.2f MHz\n", 
               pair.max_magnitude, pair.max_freq_hz / 1e6);
        printf("    Total power: %.3e\n", pair.total_power);
    }
    
    // 清理
    for (int c = 0; c < config.num_channels; c++) {
        free(channel_data[c]);
    }
    
    return true;
}

// 主函数（保持不变）
int main(int argc, char** argv) {
    print_line(70, '=');
    std::cout << "FX MULTI-CHANNEL CORRELATOR (Frequency Domain) - OPTIMIZED" << std::endl;
    print_line(70, '=');
    std::cout << std::endl;
    
    if (argc < 3) {
        printf("Usage: %s -n <files...> [n_per_frame] [num_frames] [start_sample]\n", argv[0]);
        printf("\nModes:\n");
        printf("  -n : Multi-channel mode (followed by channel files)\n");
        printf("\nParameters:\n");
        printf("  n_per_frame       : FFT points per frame (default: 1024)\n");
        printf("  num_frames        : Number of frames to accumulate (default: 1000)\n");
        printf("  start_sample      : Starting sample offset (default: 0)\n");
        printf("\nExamples:\n");
        printf("  %s -n ch1.dat ch2.dat ch3.dat\n", argv[0]);
        printf("  %s -n ch1.dat ch2.dat ch3.dat 2048 500\n", argv[0]);
        return 1;
    }
    
    // 解析模式
    int file_start_idx = 1;
    if (std::string(argv[1]) != "-n") {
        printf("Error: First argument must be -n for multi-channel mode\n");
        return 1;
    }
    file_start_idx = 2;
    
    // 收集文件列表
    std::vector<std::string> input_files;
    for (int i = file_start_idx; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find_first_not_of("0123456789") == std::string::npos) {
            break;
        }
        input_files.push_back(arg);
    }
    
    // 检查通道数
    if (input_files.size() < 2 || input_files.size() > 8) {
        printf("Error: Need 2-8 channel files, got %zu\n", input_files.size());
        return 1;
    }
    
    // 解析数字参数
    int n_per_frame = 1024;
    int num_frames = 1000;
    size_t start_sample = 0;
    
    int num_args = argc - file_start_idx - input_files.size();
    if (num_args >= 1) {
        n_per_frame = atoi(argv[file_start_idx + input_files.size()]);
    }
    if (num_args >= 2) {
        num_frames = atoi(argv[file_start_idx + input_files.size() + 1]);
    }
    if (num_args >= 3) {
        start_sample = atol(argv[file_start_idx + input_files.size() + 2]);
    }
    
    // 创建配置
    FxMultiChannelConfig config;
    config.num_channels = input_files.size();
    config.channel_files = input_files;
    for (int i = 0; i < config.num_channels; i++) {
        config.channel_names.push_back("CH" + std::to_string(i+1));
    }
    
    // 显示配置
    printf("Input files:\n");
    for (size_t i = 0; i < input_files.size(); i++) {
        printf("  CH%zu: %s\n", i+1, input_files[i].c_str());
    }
    printf("\n");
    
    printf("Processing parameters:\n");
    printf("  FFT points per frame: %d\n", n_per_frame);
    printf("  Number of frames: %d\n", num_frames);
    printf("  Total samples needed: %d\n", n_per_frame * num_frames);
    printf("  Start sample offset: %zu\n\n", start_sample);
    
    // 处理
    bool success = process_fx_multi_channel(config, n_per_frame, num_frames,
                                           0, start_sample, 100e6);
    
    if (success) {
        printf("\n✅ FX Multi-channel processing completed successfully!\n");
        return 0;
    } else {
        printf("\n❌ FX Multi-channel processing failed!\n");
        return 1;
    }
}

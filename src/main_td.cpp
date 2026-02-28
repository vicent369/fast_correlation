#include "dat_reader.h"
#include "time_domain_output.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>

// 使用dat_reader.h中定义的float2_data类型
typedef float2_data float2_td;

// 声明GPU函数
extern "C" {
    void gpu_correlate_td(float2_data* h_sig1, float2_data* h_sig2, 
                         int input_size, int fft_size,
                         float* h_result, bool normalize);
    void gpu_auto_correlate_td(float2_data* h_signal,
                              int input_size, int fft_size,
                              float* h_result, bool normalize);
    void print_gpu_info_td();
}

// 辅助函数
void print_line(int length = 60, char ch = '=') {
    std::cout << std::string(length, ch) << std::endl;
}

void print_double_line(int length = 60, char ch = '=') {
    std::cout << "\n" << std::string(length, ch) << std::endl;
}

// 找峰值函数
int find_peak(const float* data, int n) {
    int peak_idx = 0;
    float max_val = data[0];
    
    for (int i = 1; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            peak_idx = i;
        }
    }
    
    return peak_idx;
}

// 计算时延
float calculate_time_delay_td(int peak_idx, int output_size, float sample_rate_hz) {
    int max_lag = peak_idx - (output_size - 1);
    float time_delay_s = max_lag / sample_rate_hz;
    return time_delay_s * 1e9;
}

// ============ 新增：多路处理函数 ============
bool process_multi_channel(const MultiChannelConfig& config,
                          size_t chunk_size, int fft_size) {
    print_double_line(70, '=');
    std::cout << "MULTI-CHANNEL PROCESSING MODE" << std::endl;
    print_line(70, '=');
    
    printf("Number of channels: %d\n", config.num_channels);
    printf("Total pairs: %d (Auto: %d, Cross: %d)\n", 
           config.get_total_pairs(), config.num_channels, config.get_cross_pairs());
    printf("Chunk size: %zu samples\n", chunk_size);
    printf("FFT size: %d points\n\n", fft_size);
    
    // 获取文件大小（假设所有文件大小相同）
    size_t total_samples = 0;
    FILE* fp = fopen(config.channel_files[0].c_str(), "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        size_t file_size = ftell(fp);
        total_samples = file_size / 4;  // 每个样本4字节
        fclose(fp);
    }
    
    size_t num_chunks = (total_samples + chunk_size - 1) / chunk_size;
    printf("Total samples per channel: %zu\n", total_samples);
    printf("Total chunks: %zu\n\n", num_chunks);
    
    // 创建输出目录
    create_output_directory("multi_channel_results");
    
    // 生成时间戳
    time_t raw_time;
    time(&raw_time);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&raw_time));
    
    std::string prefix = "multi_channel_results/multi_" + std::string(timestamp);
    
    // 保存配置
    save_multi_channel_config(config, prefix + "_config.csv");
    
    // 初始化所有相关对的结果容器
    std::vector<CorrelationPairResult> all_pairs;
    
    // 添加自相关对
    for (int i = 0; i < config.num_channels; i++) {
        CorrelationPairResult pair;
        pair.type = AUTO_CORRELATION;
        pair.channel_i = i;
        pair.channel_j = i;
        pair.pair_name = config.channel_names[i] + "_AUTO";
        pair.cumulative_integral = 0.0;
        all_pairs.push_back(pair);
    }
    
    // 添加互相关对
    for (int i = 0; i < config.num_channels; i++) {
        for (int j = i + 1; j < config.num_channels; j++) {
            CorrelationPairResult pair;
            pair.type = CROSS_CORRELATION;
            pair.channel_i = i;
            pair.channel_j = j;
            pair.pair_name = config.channel_names[i] + "x" + config.channel_names[j];
            pair.cumulative_integral = 0.0;
            all_pairs.push_back(pair);
        }
    }
    
    // 预分配所有通道的数据缓冲区
    std::vector<float2_data*> channel_data(config.num_channels, nullptr);
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 处理每个数据块
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        printf("\nProcessing chunk %zu/%zu...\n", chunk_idx + 1, num_chunks);
        
        size_t start_sample = chunk_idx * chunk_size;
        size_t current_chunk_size = std::min(chunk_size, total_samples - start_sample);
        
        // 读取所有通道的当前块
        for (int c = 0; c < config.num_channels; c++) {
            channel_data[c] = read_dat_partial(config.channel_files[c].c_str(), 
                                               start_sample, current_chunk_size);
            if (!channel_data[c]) {
                printf("Error: Failed to read chunk %zu from channel %d\n", 
                       chunk_idx, c);
                return false;
            }
        }
        
        // ============ 修改：只按通道保存原始数据，不按相关对保存 ============
        if (chunk_idx == 0) {
            printf("\n  Saving first chunk raw signal data...\n");
            float sample_rate_hz = 100e6;
            
            // 只按通道保存原始数据（每个通道一次）
            for (int c = 0; c < config.num_channels; c++) {
                // 保存二进制格式
                std::string bin_filename = prefix + "_" + config.channel_names[c] + "_first_chunk.bin";
                save_signal_data_binary(channel_data[c], current_chunk_size, 
                                       bin_filename, chunk_idx);
                
                // 保存CSV格式
                std::string csv_filename = prefix + "_" + config.channel_names[c] + "_first_chunk.csv";
                save_signal_data_csv(channel_data[c], current_chunk_size, 
                                   csv_filename, chunk_idx, sample_rate_hz);
            }
        }
        
        // 处理所有相关对
        for (auto& pair : all_pairs) {
            float* result = (float*)malloc(fft_size * sizeof(float));
            if (!result) {
                printf("Error: Failed to allocate result memory\n");
                return false;
            }
            
            if (pair.type == AUTO_CORRELATION) {
                // 自相关
                gpu_auto_correlate_td(channel_data[pair.channel_i],
                                     current_chunk_size, fft_size,
                                     result, false);
            } else {
                // 互相关
                gpu_correlate_td(channel_data[pair.channel_i],
                                channel_data[pair.channel_j],
                                current_chunk_size, fft_size,
                                result, false);
            }
            
            // 找峰值
            int peak_idx = find_peak(result, fft_size);
            float peak_value = result[peak_idx];
            
            // 计算时延（仅互相关）
            float time_delay_ns = 0.0f;
            if (pair.type == CROSS_CORRELATION) {
                time_delay_ns = calculate_time_delay_td(peak_idx, fft_size, 100e6);
            }
            
            // 计算峰值功率
            float peak_power = peak_value * peak_value;
            pair.cumulative_integral += peak_power;
            
            // 保存结果
            pair.peak_values.push_back(peak_value);
            pair.time_delays_ns.push_back(time_delay_ns);
            pair.peak_powers.push_back(peak_power);
            pair.peak_indices.push_back(peak_idx);
            
            // 保存第一个chunk的完整相关函数
            if (chunk_idx == 0) {
                pair.first_chunk_correlation.assign(result, result + fft_size);
                pair.first_chunk_size = fft_size;
                
                // 只保存相关结果，不再保存原始信号数据
                std::string corr_bin_filename = prefix + "_" + pair.pair_name + "_first_chunk_corr.bin";
                save_correlation_data_binary(result, fft_size, corr_bin_filename, chunk_idx);
                
                std::string corr_csv_filename = prefix + "_" + pair.pair_name + "_first_chunk_corr.csv";
                save_correlation_data_csv(result, fft_size, corr_csv_filename, 
                                        chunk_idx, 100e6);
            }
            
            free(result);
            
            // 显示进度
            if (chunk_idx == num_chunks - 1) {
                printf("  %s: peak=%.2e, delay=%.1f ns\n",
                       pair.pair_name.c_str(), peak_value, time_delay_ns);
            }
        }
        
        // 清理当前块的数据
        for (int c = 0; c < config.num_channels; c++) {
            free_data(channel_data[c]);
        }
        
        // 定期保存中间结果
        if ((chunk_idx + 1) % 10 == 0 || chunk_idx == num_chunks - 1) {
            save_multi_channel_results(all_pairs, prefix + "_results_intermediate.csv");
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration<float, std::milli>(total_end - total_start);
    
    // 计算每对数据的统计信息
    for (auto& pair : all_pairs) {
        if (pair.peak_values.empty()) continue;
        
        // 峰值统计
        double sum = 0.0, sum_sq = 0.0;
        pair.max_peak = pair.peak_values[0];
        pair.min_peak = pair.peak_values[0];
        
        for (float val : pair.peak_values) {
            sum += val;
            if (val > pair.max_peak) pair.max_peak = val;
            if (val < pair.min_peak) pair.min_peak = val;
        }
        pair.mean_peak = sum / pair.peak_values.size();
        
        for (float val : pair.peak_values) {
            double diff = val - pair.mean_peak;
            sum_sq += diff * diff;
        }
        pair.std_peak = sqrt(sum_sq / pair.peak_values.size());
        
        // 时延统计（仅互相关）
        if (pair.type == CROSS_CORRELATION && !pair.time_delays_ns.empty()) {
            sum = 0.0; sum_sq = 0.0;
            pair.max_delay_ns = pair.time_delays_ns[0];
            pair.min_delay_ns = pair.time_delays_ns[0];
            
            for (float val : pair.time_delays_ns) {
                sum += val;
                if (val > pair.max_delay_ns) pair.max_delay_ns = val;
                if (val < pair.min_delay_ns) pair.min_delay_ns = val;
            }
            pair.mean_delay_ns = sum / pair.time_delays_ns.size();
            
            for (float val : pair.time_delays_ns) {
                double diff = val - pair.mean_delay_ns;
                sum_sq += diff * diff;
            }
            pair.std_delay_ns = sqrt(sum_sq / pair.time_delays_ns.size());
        }
    }
    
    // 保存最终结果
    MultiChannelStats stats;
    stats.config = config;
    stats.pairs = all_pairs;
    stats.total_time_seconds = total_duration.count() / 1000.0f;
    stats.total_samples_per_channel = total_samples;
    stats.total_chunks = num_chunks;
    stats.processing_rate_msps = total_samples / stats.total_time_seconds / 1e6;
    stats.timestamp_str = timestamp;
    
    save_multi_channel_results(all_pairs, prefix + "_results.csv");
    save_multi_channel_summary_csv(stats, prefix + "_summary.csv");
    save_multi_channel_summary_txt(stats, prefix + "_summary.txt");
    
    // 保存每对数据的第一个chunk详细数据
    for (const auto& pair : all_pairs) {
        if (!pair.first_chunk_correlation.empty()) {
            std::string pair_prefix = prefix + "_" + pair.pair_name;
            save_pair_correlation_data(pair, pair_prefix, 100e6);
        }
    }
    
    // 显示最终统计
    print_double_line(60, '=');
    std::cout << "MULTI-CHANNEL PROCESSING COMPLETE" << std::endl;
    print_line(60, '=');
    
    printf("Processing time: %.3f seconds\n", stats.total_time_seconds);
    printf("Processing rate: %.2f MS/s\n", stats.processing_rate_msps);
    printf("\nResults saved to: multi_channel_results/\n");
    
    return true;
}

// 原有的单对处理函数（保持不变，但可以调用多路处理）
bool process_full_file_td(const char* file1, const char* file2, 
                         size_t chunk_size, int fft_size) {
    // 创建单对配置
    MultiChannelConfig config;
    config.num_channels = 2;
    config.channel_files = {file1, file2};
    config.channel_names = {"CH1", "CH2"};
    
    return process_multi_channel(config, chunk_size, fft_size);
}

// 测试函数（保持不变）
bool test_small_batch_td(const char* file1, const char* file2, 
                        size_t test_size, int fft_size) {
    std::cout << "\n=== TEST MODE ===" << std::endl;
    
    float2_data* sig1_test = read_dat_partial(file1, 0, test_size);
    float2_data* sig2_test = read_dat_partial(file2, 0, test_size);
    
    if (!sig1_test || !sig2_test) {
        std::cout << "Error: Failed to read test data" << std::endl;
        return false;
    }
    
    float* result = (float*)malloc(fft_size * sizeof(float));
    if (!result) {
        std::cout << "Error: Failed to allocate result memory" << std::endl;
        free_data(sig1_test);
        free_data(sig2_test);
        return false;
    }
    
    std::cout << "Computing correlation on GPU..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    gpu_correlate_td(sig1_test, sig2_test, test_size, fft_size, result, false);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(end - start);
    
    int peak_idx = find_peak(result, fft_size);
    float sample_rate_hz = 100e6;
    float time_delay_ns = calculate_time_delay_td(peak_idx, fft_size, sample_rate_hz);
    
    printf("  GPU computation time: %.2f ms\n", duration.count());
    printf("  Peak value: %.2e at index %d\n", result[peak_idx], peak_idx);
    printf("  Time delay: %.1f ns\n", time_delay_ns);
    
    free_data(sig1_test);
    free_data(sig2_test);
    free(result);
    
    std::cout << "\n✅ Test successful!" << std::endl;
    return true;
}

int main(int argc, char** argv) {
    print_line(70, '=');
    std::cout << "GPU TIME-DOMAIN CORRELATION ANALYZER (MULTI-CHANNEL VERSION)" << std::endl;
    print_line(70, '=');
    std::cout << std::endl;
    
    // 打印GPU信息
    print_gpu_info_td();
    printf("\n");
    
    if (argc < 3) {
        printf("Usage: %s [mode] <files...> [chunk_size] [fft_size]\n", argv[0]);
        printf("\nModes:\n");
        printf("  -2 : Two-channel mode (default)\n");
        printf("  -n : Multi-channel mode (followed by channel files)\n");
        printf("\nExamples:\n");
        printf("  %s -2 a0.dat b0.dat                    # Two-channel mode\n", argv[0]);
        printf("  %s -n ch1.dat ch2.dat ch3.dat          # Three-channel mode\n", argv[0]);
        printf("  %s -n ch1.dat ch2.dat ch3.dat 1048576 1024  # With parameters\n", argv[0]);
        return 1;
    }
    
    // 解析模式
    int mode = 2;  // 默认双通道
    int file_start_idx = 1;
    
    if (std::string(argv[1]) == "-2") {
        mode = 2;
        file_start_idx = 2;
    } else if (std::string(argv[1]) == "-n") {
        mode = -1;  // 多通道模式
        file_start_idx = 2;
    }
    
    // 收集文件列表
    std::vector<std::string> input_files;
    for (int i = file_start_idx; i < argc; i++) {
        std::string arg = argv[i];
        // 检查是否为数字参数
        if (arg.find_first_not_of("0123456789") == std::string::npos) {
            break;
        }
        input_files.push_back(arg);
    }
    
    // 解析数字参数
    size_t chunk_size = 1048576;  // 默认1M
    int fft_size = 1048576;       // 默认1M
    
    int num_args = argc - file_start_idx - input_files.size();
    if (num_args >= 1) {
        chunk_size = atol(argv[file_start_idx + input_files.size()]);
    }
    if (num_args >= 2) {
        fft_size = atoi(argv[file_start_idx + input_files.size() + 1]);
    }
    
    // 显示配置
    printf("Input files:\n");
    for (size_t i = 0; i < input_files.size(); i++) {
        printf("  CH%zu: %s\n", i+1, input_files[i].c_str());
    }
    printf("\n");
    
    printf("Processing parameters:\n");
    printf("  Chunk size: %zu samples\n", chunk_size);
    printf("  FFT size: %d points\n", fft_size);
    printf("  Mode: %s\n\n", (mode == 2) ? "Two-channel" : "Multi-channel");
    
    bool success = false;
    
    if (mode == 2 && input_files.size() == 2) {
        // 双通道模式
        success = process_full_file_td(input_files[0].c_str(), 
                                       input_files[1].c_str(),
                                       chunk_size, fft_size);
    } else if (mode == -1 && input_files.size() >= 2 && input_files.size() <= 8) {
        // 多通道模式
        MultiChannelConfig config;
        config.num_channels = input_files.size();
        config.channel_files = input_files;
        
        // 生成通道名称
        for (int i = 0; i < config.num_channels; i++) {
            config.channel_names.push_back("CH" + std::to_string(i+1));
        }
        
        success = process_multi_channel(config, chunk_size, fft_size);
    } else {
        printf("Error: Invalid number of files for selected mode\n");
        printf("  Two-channel mode: need exactly 2 files\n");
        printf("  Multi-channel mode: need 2-8 files\n");
        return 1;
    }
    
    if (success) {
        printf("\n✅ Program completed successfully!\n");
        return 0;
    } else {
        printf("\n❌ Program failed!\n");
        return 1;
    }
}

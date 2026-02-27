#ifndef TIME_DOMAIN_OUTPUT_H
#define TIME_DOMAIN_OUTPUT_H

#include <string>
#include <vector>
#include <ctime>

// 直接包含dat_reader.h来使用float2_data类型
#include "dat_reader.h"

// 时域结果数据结构（单对信号）
struct TimeDomainResult {
    int chunk_id;
    int samples;
    int peak_index;
    float peak_value;
    float peak_real;
    float peak_imag;
    int lag;
    float time_delay_ns;
    float peak_power;
    time_t timestamp;
};

// 统计数据结构（单对信号）
struct TimeDomainStats {
    double total_time_seconds;
    size_t total_samples;
    int total_chunks;
    double processing_rate_msps;
    
    double mean_delay_ns;
    double std_delay_ns;
    double min_delay_ns;
    double max_delay_ns;
    
    double mean_correlation;
    double std_correlation;
    double max_correlation;
    double min_correlation;
    
    double cumulative_integral;
    double avg_peak_power;
    double max_peak_power;
    double min_peak_power;
    
    std::string file1;
    std::string file2;
    float sample_rate_hz;
    std::string timestamp_str;
};

// ============ 新增：多路数据处理结构 ============

// 相关类型枚举
enum CorrelationType {
    AUTO_CORRELATION,  // 自相关
    CROSS_CORRELATION  // 互相关
};

// 单对相关结果（自相关或互相关）
struct CorrelationPairResult {
    CorrelationType type;
    int channel_i;               // 通道i索引 (0-based)
    int channel_j;               // 通道j索引 (0-based, 自相关时 i=j)
    std::string pair_name;       // 如 "CH1_AUTO", "CH1xCH2"
    
    // 每个chunk的结果
    std::vector<float> peak_values;
    std::vector<float> time_delays_ns;  // 互相关有时延，自相关时延为0
    std::vector<float> peak_powers;
    std::vector<int> peak_indices;
    
    // 累积积分
    double cumulative_integral;
    
    // 第一个chunk的完整相关函数（用于详细分析）
    std::vector<float> first_chunk_correlation;
    int first_chunk_size;
    
    // 统计信息
    double mean_peak;
    double std_peak;
    double max_peak;
    double min_peak;
    
    double mean_delay_ns;
    double std_delay_ns;
    double min_delay_ns;
    double max_delay_ns;
};

// 多路配置结构
struct MultiChannelConfig {
    int num_channels;                       // 通道数 (1-8)
    std::vector<std::string> channel_files; // 每个通道的文件名
    std::vector<std::string> channel_names; // 通道名称 (如 "CH1", "CH2"等)
    
    MultiChannelConfig() : num_channels(0) {}
    
    // 获取总的相关对数量
    int get_total_pairs() const {
        if (num_channels <= 0) return 0;
        // 自相关: n 个, 互相关: n*(n-1)/2 对
        return num_channels + num_channels * (num_channels - 1) / 2;
    }
    
    // 获取互相关对数量
    int get_cross_pairs() const {
        return num_channels * (num_channels - 1) / 2;
    }
};

// 多路结果统计
struct MultiChannelStats {
    MultiChannelConfig config;
    std::vector<CorrelationPairResult> pairs;
    
    // 处理信息
    float total_time_seconds;
    size_t total_samples_per_channel;  // 每个通道的样本数（假设所有通道相同）
    int total_chunks;
    float processing_rate_msps;
    std::string timestamp_str;
    
    // 获取指定对的索引
    int get_pair_index(int i, int j) const {
        if (i == j) {
            // 自相关：CH{i} 在列表开头
            return i;
        } else {
            // 互相关：按顺序排列
            int idx = config.num_channels;  // 跳过所有自相关
            for (int a = 0; a < i; a++) {
                for (int b = a + 1; b < config.num_channels; b++) {
                    if (a == i && b == j) return idx;
                    idx++;
                }
            }
            return -1;
        }
    }
};

// ============ 新增：多路输出函数声明 ============

// 保存多路配置信息
bool save_multi_channel_config(const MultiChannelConfig& config,
                               const std::string& filename);

// 保存所有相关对的结果
bool save_multi_channel_results(const std::vector<CorrelationPairResult>& pairs,
                               const std::string& filename);

// 保存多路统计摘要（CSV格式）
bool save_multi_channel_summary_csv(const MultiChannelStats& stats,
                                   const std::string& filename);

// 保存多路统计摘要（文本格式）
bool save_multi_channel_summary_txt(const MultiChannelStats& stats,
                                   const std::string& filename);

// 保存单个相关对的详细数据（第一个chunk）
bool save_pair_correlation_data(const CorrelationPairResult& pair,
                               const std::string& filename_prefix,
                               float sample_rate_hz);

// ============ 原有函数声明（保持不变） ============

// 文件输出函数（单对信号）
bool save_time_domain_results(const std::vector<TimeDomainResult>& results,
                             const std::string& filename);

bool save_time_domain_summary_csv(const TimeDomainStats& stats,
                                 const std::string& filename);

bool save_time_domain_summary_txt(const TimeDomainStats& stats,
                                 const std::string& filename);

bool save_correlation_data_binary(const float* correlation_data, int n,
                                 const std::string& filename, int chunk_id = 0);

bool save_correlation_data_csv(const float* correlation_data, int n,
                              const std::string& filename, int chunk_id,
                              float sample_rate_hz);

// 信号数据保存函数
bool save_signal_data_binary(const float2_data* signal_data, int n,
                            const std::string& filename, int chunk_id = 0);

bool save_signal_data_csv(const float2_data* signal_data, int n,
                         const std::string& filename, int chunk_id,
                         float sample_rate_hz);

// 工具函数
std::string extract_base_filename(const std::string& fullpath);
void create_output_directory(const std::string& dirname);

#endif

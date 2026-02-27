#include "time_domain_output.h"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <cstdlib>
#include <sstream>

// 提取基本文件名（去掉路径和扩展名）
std::string extract_base_filename(const std::string& fullpath) {
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

// 创建输出目录
void create_output_directory(const std::string& dirname) {
    struct stat st = {0};
    if (stat(dirname.c_str(), &st) == -1) {
        int result = system(("mkdir -p " + dirname).c_str());
        if (result != 0) {
            printf("Warning: Failed to create directory %s\n", dirname.c_str());
        } else {
            printf("Created output directory: %s\n", dirname.c_str());
        }
    }
}

// 原有函数：保存时域结果（单对信号）
bool save_time_domain_results(const std::vector<TimeDomainResult>& results,
                             const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    fprintf(fp, "chunk_id,samples,peak_index,peak_value,peak_real,peak_imag,max_lag,time_delay_ns,peak_power,timestamp\n");
    
    for (const auto& r : results) {
        fprintf(fp, "%d,%d,%d,%.6e,%.6e,%.6e,%d,%.3f,%.6e,%ld\n",
                r.chunk_id, r.samples, r.peak_index,
                r.peak_value, r.peak_real, r.peak_imag,
                r.lag, r.time_delay_ns, r.peak_power, r.timestamp);
    }
    
    fclose(fp);
    printf("  ✓ Saved time-domain results to: %s (%zu records)\n", 
           filename.c_str(), results.size());
    return true;
}

// 原有函数：保存统计摘要CSV（单对信号）
bool save_time_domain_summary_csv(const TimeDomainStats& stats,
                                 const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    fprintf(fp, "parameter,value,unit\n");
    fprintf(fp, "file1,%s,\n", stats.file1.c_str());
    fprintf(fp, "file2,%s,\n", stats.file2.c_str());
    fprintf(fp, "sample_rate_hz,%.0f,Hz\n", stats.sample_rate_hz);
    fprintf(fp, "total_samples,%zu,\n", stats.total_samples);
    fprintf(fp, "total_chunks,%d,\n", stats.total_chunks);
    fprintf(fp, "total_time_seconds,%.3f,s\n", stats.total_time_seconds);
    fprintf(fp, "processing_rate_msps,%.2f,MS/s\n", stats.processing_rate_msps);
    fprintf(fp, "data_throughput,%.1f,MB/s\n", stats.processing_rate_msps * 8.0f);
    
    fprintf(fp, "mean_delay_ns,%.3f,ns\n", stats.mean_delay_ns);
    fprintf(fp, "std_delay_ns,%.3f,ns\n", stats.std_delay_ns);
    fprintf(fp, "min_delay_ns,%.3f,ns\n", stats.min_delay_ns);
    fprintf(fp, "max_delay_ns,%.3f,ns\n", stats.max_delay_ns);
    
    fprintf(fp, "mean_correlation,%.3f,\n", stats.mean_correlation);
    fprintf(fp, "std_correlation,%.3f,\n", stats.std_correlation);
    fprintf(fp, "max_correlation,%.3f,\n", stats.max_correlation);
    fprintf(fp, "min_correlation,%.3f,\n", stats.min_correlation);
    
    fprintf(fp, "cumulative_integral,%.6e,\n", stats.cumulative_integral);
    fprintf(fp, "avg_peak_power,%.6e,\n", stats.avg_peak_power);
    fprintf(fp, "max_peak_power,%.6e,\n", stats.max_peak_power);
    fprintf(fp, "min_peak_power,%.6e,\n", stats.min_peak_power);
    
    fprintf(fp, "timestamp,%s,\n", stats.timestamp_str.c_str());
    
    fclose(fp);
    printf("  ✓ Saved summary (CSV) to: %s\n", filename.c_str());
    return true;
}

// 原有函数：保存统计摘要TXT（单对信号）
bool save_time_domain_summary_txt(const TimeDomainStats& stats,
                                 const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    fprintf(fp, "===================================================\n");
    fprintf(fp, "      TIME DOMAIN CORRELATION ANALYSIS SUMMARY\n");
    fprintf(fp, "      (GPU Accelerated with Full File Output)\n");
    fprintf(fp, "===================================================\n\n");
    
    fprintf(fp, "INPUT FILES:\n");
    fprintf(fp, "  File 1: %s\n", stats.file1.c_str());
    fprintf(fp, "  File 2: %s\n\n", stats.file2.c_str());
    
    fprintf(fp, "PROCESSING INFORMATION:\n");
    fprintf(fp, "  Sample rate: %.1f MS/s\n", stats.sample_rate_hz / 1e6);
    fprintf(fp, "  Total samples: %zu (%.1f MB per file)\n", 
            stats.total_samples, stats.total_samples * 8.0 / 1024.0 / 1024.0);
    fprintf(fp, "  Total chunks: %d\n", stats.total_chunks);
    fprintf(fp, "  Total processing time: %.3f seconds\n", stats.total_time_seconds);
    fprintf(fp, "  Processing rate: %.2f MS/s\n", stats.processing_rate_msps);
    fprintf(fp, "  Data throughput: %.1f MB/s\n\n", stats.processing_rate_msps * 8.0f);
    
    fprintf(fp, "TIME DELAY STATISTICS:\n");
    fprintf(fp, "  Mean time delay: %.3f ns\n", stats.mean_delay_ns);
    fprintf(fp, "  Standard deviation: %.3f ns\n", stats.std_delay_ns);
    fprintf(fp, "  Minimum delay: %.3f ns\n", stats.min_delay_ns);
    fprintf(fp, "  Maximum delay: %.3f ns\n", stats.max_delay_ns);
    fprintf(fp, "  Delay range: %.3f ns\n\n", stats.max_delay_ns - stats.min_delay_ns);
    
    fprintf(fp, "CORRELATION PEAK STATISTICS:\n");
    fprintf(fp, "  Mean peak correlation: %.3f\n", stats.mean_correlation);
    fprintf(fp, "  Std peak correlation: %.3f\n", stats.std_correlation);
    fprintf(fp, "  Maximum peak correlation: %.3f\n", stats.max_correlation);
    fprintf(fp, "  Minimum peak correlation: %.3f\n", stats.min_correlation);
    fprintf(fp, "  Peak SNR estimate: %.1f (ratio)\n\n", 
            stats.mean_correlation / std::max(1e-10, stats.std_correlation));
    
    fprintf(fp, "INTEGRATION RESULTS:\n");
    fprintf(fp, "  Cumulative integral: %.6e\n", stats.cumulative_integral);
    fprintf(fp, "  Average peak power: %.6e\n", stats.avg_peak_power);
    fprintf(fp, "  Maximum peak power: %.6e\n", stats.max_peak_power);
    fprintf(fp, "  Minimum peak power: %.6e\n\n", stats.min_peak_power);
    
    fprintf(fp, "OUTPUT FILES:\n");
    fprintf(fp, "  - results.csv: Detailed chunk-by-chunk data\n");
    fprintf(fp, "  - summary.csv: Statistical summary (CSV format)\n");
    fprintf(fp, "  - summary.txt: This human-readable summary\n");
    fprintf(fp, "  - first_chunk_correlation.bin: Raw correlation data\n");
    fprintf(fp, "  - first_chunk_correlation.csv: Correlation data in CSV\n");
    fprintf(fp, "  - first_chunk_signal1.bin: Raw signal 1 data (first chunk)\n");
    fprintf(fp, "  - first_chunk_signal1.csv: Signal 1 data in CSV (first chunk)\n");
    fprintf(fp, "  - first_chunk_signal2.bin: Raw signal 2 data (first chunk)\n");
    fprintf(fp, "  - first_chunk_signal2.csv: Signal 2 data in CSV (first chunk)\n");
    fprintf(fp, "  - intermediate.csv: Periodic backup during processing\n\n");
    
    fprintf(fp, "TIMESTAMP: %s\n", stats.timestamp_str.c_str());
    
    fclose(fp);
    printf("  ✓ Saved summary (text) to: %s\n", filename.c_str());
    return true;
}

// 原有函数：保存相关数据二进制
bool save_correlation_data_binary(const float* correlation_data, int n,
                                 const std::string& filename, int chunk_id) {
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    size_t written = fwrite(correlation_data, sizeof(float), n, fp);
    fclose(fp);
    
    if (written == n) {
        printf("  ✓ Saved correlation data (chunk %d) to: %s (%d samples, %.1f MB)\n",
               chunk_id, filename.c_str(), n, n * sizeof(float) / 1024.0 / 1024.0);
        return true;
    } else {
        printf("  ✗ Error writing correlation data to: %s (wrote %zu of %d)\n",
               filename.c_str(), written, n);
        return false;
    }
}

// 原有函数：保存相关数据CSV
bool save_correlation_data_csv(const float* correlation_data, int n,
                              const std::string& filename, int chunk_id,
                              float sample_rate_hz) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    fprintf(fp, "index,lag_samples,time_delay_ns,correlation_value\n");
    
    for (int i = 0; i < n; i++) {
        int lag = i - (n - 1);
        float time_delay_ns = lag / sample_rate_hz * 1e9;
        fprintf(fp, "%d,%d,%.3f,%.6e\n", 
                i, lag, time_delay_ns, correlation_data[i]);
    }
    
    fclose(fp);
    printf("  ✓ Saved correlation CSV (chunk %d) to: %s\n", 
           chunk_id, filename.c_str());
    return true;
}

// 原有函数：保存信号数据二进制
bool save_signal_data_binary(const float2_data* signal_data, int n,
                            const std::string& filename, int chunk_id) {
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    size_t written = fwrite(signal_data, sizeof(float2_data), n, fp);
    fclose(fp);
    
    if (written == n) {
        printf("  ✓ Saved signal data (chunk %d) to: %s (%d complex samples, %.1f MB)\n",
               chunk_id, filename.c_str(), n, n * sizeof(float2_data) / 1024.0 / 1024.0);
        return true;
    } else {
        printf("  ✗ Error writing signal data to: %s (wrote %zu of %d)\n",
               filename.c_str(), written, n);
        return false;
    }
}

// 原有函数：保存信号数据CSV
bool save_signal_data_csv(const float2_data* signal_data, int n,
                         const std::string& filename, int chunk_id,
                         float sample_rate_hz) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    fprintf(fp, "index,time_ns,real_part,imag_part,magnitude,phase_deg\n");
    
    for (int i = 0; i < n; i++) {
        float real_part = signal_data[i].x;
        float imag_part = signal_data[i].y;
        float magnitude = sqrtf(real_part * real_part + imag_part * imag_part);
        float phase_deg = atan2f(imag_part, real_part) * 180.0f / M_PI;
        float time_ns = i / sample_rate_hz * 1e9;
        
        fprintf(fp, "%d,%.3f,%.6e,%.6e,%.6e,%.2f\n", 
                i, time_ns, real_part, imag_part, magnitude, phase_deg);
    }
    
    fclose(fp);
    printf("  ✓ Saved signal CSV (chunk %d) to: %s (%d complex samples)\n", 
           chunk_id, filename.c_str(), n);
    return true;
}

// ============ 新增：多路输出函数实现 ============

// 保存多路配置信息
bool save_multi_channel_config(const MultiChannelConfig& config,
                               const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    fprintf(fp, "# Multi-Channel Correlation Configuration\n");
    fprintf(fp, "num_channels,%d\n", config.num_channels);
    fprintf(fp, "total_pairs,%d (auto: %d, cross: %d)\n", 
            config.get_total_pairs(), config.num_channels, config.get_cross_pairs());
    fprintf(fp, "\nchannel_index,channel_name,file_path\n");
    
    for (int i = 0; i < config.num_channels; i++) {
        fprintf(fp, "%d,%s,%s\n", i, 
                config.channel_names[i].c_str(),
                config.channel_files[i].c_str());
    }
    
    fclose(fp);
    printf("  ✓ Saved multi-channel config to: %s\n", filename.c_str());
    return true;
}

// 保存所有相关对的结果
bool save_multi_channel_results(const std::vector<CorrelationPairResult>& pairs,
                               const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    // 写入CSV头部
    fprintf(fp, "pair_name,type,chunk_id,peak_value,time_delay_ns,peak_power,peak_index\n");
    
    // 写入每对数据的每个chunk
    for (const auto& pair : pairs) {
        const char* type_str = (pair.type == AUTO_CORRELATION) ? "AUTO" : "CROSS";
        
        for (size_t i = 0; i < pair.peak_values.size(); i++) {
            fprintf(fp, "%s,%s,%zu,%.6e,%.3f,%.6e,%d\n",
                    pair.pair_name.c_str(),
                    type_str,
                    i,
                    pair.peak_values[i],
                    pair.time_delays_ns[i],
                    pair.peak_powers[i],
                    pair.peak_indices[i]);
        }
    }
    
    fclose(fp);
    printf("  ✓ Saved multi-channel results to: %s\n", filename.c_str());
    return true;
}

// 保存多路统计摘要（CSV格式）- 更详细的版本
bool save_multi_channel_summary_csv(const MultiChannelStats& stats,
                                   const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    // 基本信息
    fprintf(fp, "# MULTI-CHANNEL CORRELATION SUMMARY\n");
    fprintf(fp, "# Generated: %s\n", stats.timestamp_str.c_str());
    fprintf(fp, "# Number of channels: %d\n", stats.config.num_channels);
    fprintf(fp, "# Total pairs: %d\n", (int)stats.pairs.size());
    fprintf(fp, "#\n");
    
    // 处理信息
    fprintf(fp, "Category,Parameter,Value,Unit\n");
    fprintf(fp, "Processing,Timestamp,%s,\n", stats.timestamp_str.c_str());
    fprintf(fp, "Processing,NumChannels,%d,\n", stats.config.num_channels);
    fprintf(fp, "Processing,NumPairs,%d,\n", (int)stats.pairs.size());
    fprintf(fp, "Processing,SamplesPerChannel,%zu,\n", stats.total_samples_per_channel);
    fprintf(fp, "Processing,TotalChunks,%d,\n", stats.total_chunks);
    fprintf(fp, "Processing,ProcessingTime,%.3f,s\n", stats.total_time_seconds);
    fprintf(fp, "Processing,ProcessingRate,%.2f,MS/s\n", stats.processing_rate_msps);
    fprintf(fp, "Processing,DataThroughput,%.2f,MB/s\n", stats.processing_rate_msps * 8.0f);
    fprintf(fp, "\n");
    
    // 通道信息
    fprintf(fp, "# Channel Configuration\n");
    fprintf(fp, "ChannelIndex,ChannelName,FilePath\n");
    for (int i = 0; i < stats.config.num_channels; i++) {
        fprintf(fp, "%d,%s,%s\n", i, 
                stats.config.channel_names[i].c_str(),
                stats.config.channel_files[i].c_str());
    }
    fprintf(fp, "\n");
    
    // 每对数据的详细统计
    fprintf(fp, "# Pair Statistics\n");
    fprintf(fp, "PairName,Type,Channel_i,Channel_j,"
                "MeanPeak,StdPeak,MaxPeak,MinPeak,"
                "MeanDelay_ns,StdDelay_ns,MinDelay_ns,MaxDelay_ns,"
                "CumulativeIntegral,NumChunks\n");
    
    for (const auto& pair : stats.pairs) {
        const char* type_str = (pair.type == AUTO_CORRELATION) ? "AUTO" : "CROSS";
        
        fprintf(fp, "%s,%s,%d,%d,"
                    "%.6e,%.6e,%.6e,%.6e,"
                    "%.3f,%.3f,%.3f,%.3f,"
                    "%.6e,%zu\n",
                pair.pair_name.c_str(),
                type_str,
                pair.channel_i,
                pair.channel_j,
                pair.mean_peak,
                pair.std_peak,
                pair.max_peak,
                pair.min_peak,
                pair.mean_delay_ns,
                pair.std_delay_ns,
                pair.min_delay_ns,
                pair.max_delay_ns,
                pair.cumulative_integral,
                pair.peak_values.size());
    }
    fprintf(fp, "\n");
    
    // 输出文件列表
    fprintf(fp, "# Generated Files\n");
    fprintf(fp, "FileType,FileName,Description\n");
    
    std::string prefix = filename.substr(0, filename.find_last_of("_"));
    prefix = prefix.substr(0, prefix.find_last_of("_"));
    
    fprintf(fp, "Configuration,%s_config.csv,Channel configuration\n", prefix.c_str());
    fprintf(fp, "Results,%s_results.csv,Detailed chunk results\n", prefix.c_str());
    fprintf(fp, "SummaryCSV,%s_summary.csv,This summary file\n", prefix.c_str());
    fprintf(fp, "SummaryText,%s_summary.txt,Human-readable summary\n", prefix.c_str());
    
    for (int i = 0; i < stats.config.num_channels; i++) {
        fprintf(fp, "RawSignalBinary,%s_%s_first_chunk.bin,Raw signal data for %s\n",
                prefix.c_str(), stats.config.channel_names[i].c_str(),
                stats.config.channel_names[i].c_str());
        fprintf(fp, "RawSignalCSV,%s_%s_first_chunk.csv,Raw signal CSV for %s\n",
                prefix.c_str(), stats.config.channel_names[i].c_str(),
                stats.config.channel_names[i].c_str());
    }
    
    for (const auto& pair : stats.pairs) {
        fprintf(fp, "CorrelationBinary,%s_%s_first_chunk_corr.bin,Correlation data for %s\n",
                prefix.c_str(), pair.pair_name.c_str(), pair.pair_name.c_str());
        fprintf(fp, "CorrelationCSV,%s_%s_first_chunk_corr.csv,Correlation CSV for %s\n",
                prefix.c_str(), pair.pair_name.c_str(), pair.pair_name.c_str());
    }
    
    fclose(fp);
    printf("  ✓ Saved detailed multi-channel summary (CSV) to: %s\n", filename.c_str());
    return true;
}

// 保存多路统计摘要（文本格式）- 更详细的版本
bool save_multi_channel_summary_txt(const MultiChannelStats& stats,
                                   const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "            MULTI-CHANNEL CORRELATION ANALYSIS SUMMARY\n");
    fprintf(fp, "            (GPU Accelerated with Full File Output)\n");
    fprintf(fp, "================================================================================\n\n");
    
    // ========== 处理信息 ==========
    fprintf(fp, "1. PROCESSING INFORMATION\n");
    fprintf(fp, "------------------------\n");
    fprintf(fp, "  Timestamp            : %s\n", stats.timestamp_str.c_str());
    fprintf(fp, "  Number of channels   : %d\n", stats.config.num_channels);
    fprintf(fp, "  Total pairs analyzed : %d (Auto: %d, Cross: %d)\n", 
            (int)stats.pairs.size(), stats.config.num_channels, stats.config.get_cross_pairs());
    fprintf(fp, "  Samples per channel  : %zu (%.2f MB)\n", 
            stats.total_samples_per_channel,
            stats.total_samples_per_channel * 8.0 / 1024.0 / 1024.0);
    fprintf(fp, "  Total chunks         : %d\n", stats.total_chunks);
    fprintf(fp, "  Processing time      : %.3f seconds\n", stats.total_time_seconds);
    fprintf(fp, "  Processing rate      : %.2f MS/s\n", stats.processing_rate_msps);
    fprintf(fp, "  Data throughput      : %.2f MB/s\n\n", stats.processing_rate_msps * 8.0f);
    
    // ========== 通道配置 ==========
    fprintf(fp, "2. CHANNEL CONFIGURATION\n");
    fprintf(fp, "------------------------\n");
    for (int i = 0; i < stats.config.num_channels; i++) {
        fprintf(fp, "  CH%d:\n", i+1);
        fprintf(fp, "      Name: %s\n", stats.config.channel_names[i].c_str());
        fprintf(fp, "      File: %s\n", stats.config.channel_files[i].c_str());
    }
    fprintf(fp, "\n");
    
    // ========== 每对数据的详细统计 ==========
    fprintf(fp, "3. PAIR STATISTICS\n");
    fprintf(fp, "------------------\n");
    
    for (const auto& pair : stats.pairs) {
        fprintf(fp, "  %s [%s]:\n", pair.pair_name.c_str(),
                (pair.type == AUTO_CORRELATION) ? "AUTO-CORRELATION" : "CROSS-CORRELATION");
        fprintf(fp, "      Channels          : CH%d", pair.channel_i + 1);
        if (pair.type == CROSS_CORRELATION) {
            fprintf(fp, " x CH%d\n", pair.channel_j + 1);
        } else {
            fprintf(fp, " (self)\n");
        }
        
        fprintf(fp, "      Peak Correlation:\n");
        fprintf(fp, "          Mean value    : %.3e\n", pair.mean_peak);
        fprintf(fp, "          Std deviation : %.3e\n", pair.std_peak);
        fprintf(fp, "          Maximum       : %.3e\n", pair.max_peak);
        fprintf(fp, "          Minimum       : %.3e\n", pair.min_peak);
        fprintf(fp, "          SNR estimate  : %.1f\n", 
                pair.mean_peak / std::max(1e-10f, (float)pair.std_peak));
        
        if (pair.type == CROSS_CORRELATION) {
            fprintf(fp, "      Time Delay:\n");
            fprintf(fp, "          Mean value    : %.3f ns\n", pair.mean_delay_ns);
            fprintf(fp, "          Std deviation : %.3f ns\n", pair.std_delay_ns);
            fprintf(fp, "          Minimum       : %.3f ns\n", pair.min_delay_ns);
            fprintf(fp, "          Maximum       : %.3f ns\n", pair.max_delay_ns);
            fprintf(fp, "          Range         : %.3f ns\n", 
                    pair.max_delay_ns - pair.min_delay_ns);
        } else {
            fprintf(fp, "      Time Delay        : 0 ns (auto-correlation)\n");
        }
        
        fprintf(fp, "      Peak Power:\n");
        fprintf(fp, "          Cumulative integral : %.6e\n", pair.cumulative_integral);
        
        // 计算平均功率
        double avg_power = 0.0;
        for (float power : pair.peak_powers) {
            avg_power += power;
        }
        avg_power /= pair.peak_powers.size();
        fprintf(fp, "          Average power      : %.6e\n", avg_power);
        
        fprintf(fp, "      Chunks processed   : %zu\n\n", pair.peak_values.size());
    }
    
    // ========== 输出文件列表 ==========
    fprintf(fp, "4. OUTPUT FILES\n");
    fprintf(fp, "------------------\n");
    
    // 获取文件名前缀
    std::string prefix = filename.substr(0, filename.find_last_of("_"));
    prefix = prefix.substr(0, prefix.find_last_of("_"));
    
    fprintf(fp, "\n  4.1 Configuration Files:\n");
    fprintf(fp, "      %s_config.csv\n", prefix.c_str());
    fprintf(fp, "          Description: Channel configuration information\n");
    fprintf(fp, "          Format: CSV\n");
    fprintf(fp, "          Contents: Channel index, channel name, file path\n\n");
    
    fprintf(fp, "  4.2 Result Files:\n");
    fprintf(fp, "      %s_results.csv\n", prefix.c_str());
    fprintf(fp, "          Description: Detailed chunk-by-chunk results for all pairs\n");
    fprintf(fp, "          Format: CSV\n");
    fprintf(fp, "          Contents: pair_name, type, chunk_id, peak_value, time_delay_ns, peak_power, peak_index\n\n");
    
    fprintf(fp, "  4.3 Summary Files:\n");
    fprintf(fp, "      %s_summary.csv\n", prefix.c_str());
    fprintf(fp, "          Description: Statistical summary in CSV format\n");
    fprintf(fp, "          Format: CSV\n");
    fprintf(fp, "          Contents: Pair, Type, Mean Peak, Std Peak, Max Peak, Min Peak, Mean Delay, Std Delay, etc.\n\n");
    
    fprintf(fp, "      %s_summary.txt\n", prefix.c_str());
    fprintf(fp, "          Description: This human-readable summary file\n");
    fprintf(fp, "          Format: Text\n");
    fprintf(fp, "          Contents: Complete processing information and statistics\n\n");
    
    fprintf(fp, "  4.4 Raw Signal Data (First Chunk):\n");
    for (int i = 0; i < stats.config.num_channels; i++) {
        fprintf(fp, "      %s_%s_first_chunk.bin\n", 
                prefix.c_str(), stats.config.channel_names[i].c_str());
        fprintf(fp, "          Description: Raw signal data for channel %s (first chunk only)\n",
                stats.config.channel_names[i].c_str());
        fprintf(fp, "          Format: Binary (float2_data structure)\n");
        fprintf(fp, "          Size: %zu samples (%.1f MB)\n", 
                stats.total_samples_per_channel > stats.total_chunks ? 
                stats.total_samples_per_channel / stats.total_chunks : 
                stats.total_samples_per_channel,
                (stats.total_samples_per_channel > stats.total_chunks ? 
                (stats.total_samples_per_channel / stats.total_chunks) * sizeof(float2_data) / 1024.0 / 1024.0 : 
                stats.total_samples_per_channel * sizeof(float2_data) / 1024.0 / 1024.0));
        
        fprintf(fp, "      %s_%s_first_chunk.csv\n", 
                prefix.c_str(), stats.config.channel_names[i].c_str());
        fprintf(fp, "          Description: Raw signal data in CSV format\n");
        fprintf(fp, "          Format: CSV\n");
        fprintf(fp, "          Columns: index, time_ns, real_part, imag_part, magnitude, phase_deg\n\n");
    }
    
    fprintf(fp, "  4.5 Correlation Data (First Chunk):\n");
    for (const auto& pair : stats.pairs) {
        fprintf(fp, "      %s_%s_first_chunk_corr.bin\n", 
                prefix.c_str(), pair.pair_name.c_str());
        fprintf(fp, "          Description: Complete correlation function for %s (first chunk only)\n",
                pair.pair_name.c_str());
        fprintf(fp, "          Format: Binary (float array)\n");
        fprintf(fp, "          Size: %d points (%.1f MB)\n", 
                pair.first_chunk_size,
                pair.first_chunk_size * sizeof(float) / 1024.0 / 1024.0);
        
        fprintf(fp, "      %s_%s_first_chunk_corr.csv\n", 
                prefix.c_str(), pair.pair_name.c_str());
        fprintf(fp, "          Description: Complete correlation function in CSV format\n");
        fprintf(fp, "          Format: CSV\n");
        fprintf(fp, "          Columns: index, lag_samples, time_delay_ns, correlation_value\n\n");
    }
    
    // ========== 文件统计 ==========
    fprintf(fp, "5. FILE STATISTICS\n");
    fprintf(fp, "------------------\n");
    
    int total_files = 2 +  // config.csv, results.csv
                      2 +  // summary.csv, summary.txt
                      2 * stats.config.num_channels +  // signal bin + csv per channel
                      2 * stats.pairs.size();  // correlation bin + csv per pair
    
    fprintf(fp, "  Total files generated : %d\n", total_files);
    fprintf(fp, "  Configuration files   : 2\n");
    fprintf(fp, "  Result files          : 1\n");
    fprintf(fp, "  Summary files         : 2\n");
    fprintf(fp, "  Raw signal files      : %d (%d channels × 2 formats)\n", 
            2 * stats.config.num_channels, stats.config.num_channels);
    fprintf(fp, "  Correlation files     : %d (%d pairs × 2 formats)\n", 
            2 * (int)stats.pairs.size(), (int)stats.pairs.size());
    
    fprintf(fp, "\n  All files are saved in the 'multi_channel_results/' directory\n");
    fprintf(fp, "  File naming convention: [timestamp]_[description].[ext]\n\n");
    
    // ========== 使用建议 ==========
    fprintf(fp, "6. USAGE NOTES\n");
    fprintf(fp, "------------------\n");
    fprintf(fp, "  - The first chunk data can be used for detailed analysis and plotting\n");
    fprintf(fp, "  - Binary files are suitable for fast loading in MATLAB/Python\n");
    fprintf(fp, "  - CSV files can be opened in Excel or other spreadsheet software\n");
    fprintf(fp, "  - The cumulative integral represents integrated power over all chunks\n");
    fprintf(fp, "  - Time delays are calculated assuming 100 MHz sampling rate\n\n");
    
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "END OF SUMMARY\n");
    fprintf(fp, "================================================================================\n");
    
    fclose(fp);
    printf("  ✓ Saved detailed multi-channel summary to: %s\n", filename.c_str());
    return true;
}

// 保存单个相关对的详细数据（第一个chunk）
// 在 time_domain_output.cpp 中修改这个函数
bool save_pair_correlation_data(const CorrelationPairResult& pair,
                               const std::string& filename_prefix,
                               float sample_rate_hz) {
    if (pair.first_chunk_correlation.empty()) {
        return false;
    }
    
    // 只保存相关数据，不保存原始信号数据
    std::string corr_bin_filename = filename_prefix + "_first_chunk_corr.bin";
    FILE* fp_bin = fopen(corr_bin_filename.c_str(), "wb");
    if (fp_bin) {
        fwrite(pair.first_chunk_correlation.data(), sizeof(float), 
               pair.first_chunk_correlation.size(), fp_bin);
        fclose(fp_bin);
    }
    
    std::string corr_csv_filename = filename_prefix + "_first_chunk_corr.csv";
    FILE* fp_csv = fopen(corr_csv_filename.c_str(), "w");
    if (fp_csv) {
        fprintf(fp_csv, "index,lag_samples,time_delay_ns,correlation_value\n");
        int n = pair.first_chunk_correlation.size();
        for (int i = 0; i < n; i++) {
            int lag = i - (n - 1);
            float time_delay_ns = lag / sample_rate_hz * 1e9;
            fprintf(fp_csv, "%d,%d,%.3f,%.6e\n", 
                    i, lag, time_delay_ns, pair.first_chunk_correlation[i]);
        }
        fclose(fp_csv);
    }
    
    printf("  ✓ Saved first chunk correlation data for %s\n", 
           filename_prefix.c_str());
    return true;
}

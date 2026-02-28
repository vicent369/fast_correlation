#include "fx_correlator.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <sys/stat.h>

// 创建输出目录
void create_fx_output_directory(const std::string& dirname) {
    struct stat st = {0};
    if (stat(dirname.c_str(), &st) == -1) {
        mkdir(dirname.c_str(), 0755);
    }
}

// 提取基本文件名
std::string get_fx_base_filename(const std::string& fullpath) {
    std::string filename = fullpath;
    
    size_t slash_pos = filename.find_last_of("/\\");
    if (slash_pos != std::string::npos) {
        filename = filename.substr(slash_pos + 1);
    }
    
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos != std::string::npos) {
        filename = filename.substr(0, dot_pos);
    }
    
    return filename;
}

// 保存复数谱到CSV
bool save_fx_complex_spectrum_csv(const complex_t* data, int n, 
                                 float sample_rate_hz,
                                 const std::string& filename,
                                 const std::string& pair_name) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    fprintf(fp, "# Pair: %s\n", pair_name.c_str());
    fprintf(fp, "frequency_index,frequency_hz,real_part,imag_part,magnitude,phase_deg\n");
    
    for (int i = 0; i < n; i++) {
        float real = data[i].x;
        float imag = data[i].y;
        float magnitude = sqrtf(real*real + imag*imag);
        float phase_deg = atan2f(imag, real) * 180.0f / M_PI;
        
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
    return true;
}

// 保存二进制文件
bool save_fx_complex_spectrum_binary(const complex_t* data, int n,
                                    const std::string& filename,
                                    const std::string& pair_name) {
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    fwrite(data, sizeof(complex_t), n, fp);
    fclose(fp);
    
    return true;
}

// 保存多通道配置
bool save_fx_multi_channel_config(const FxMultiChannelConfig& config,
                                  const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    fprintf(fp, "# FX Multi-Channel Correlation Configuration\n");
    fprintf(fp, "num_channels,%d\n", config.num_channels);
    fprintf(fp, "total_pairs,%d (auto: %d, cross: %d)\n", 
            config.get_total_pairs(), config.num_channels, 
            config.num_channels * (config.num_channels - 1) / 2);
    fprintf(fp, "\nchannel_index,channel_name,file_path\n");
    
    for (int i = 0; i < config.num_channels; i++) {
        fprintf(fp, "%d,%s,%s\n", i, 
                config.channel_names[i].c_str(),
                config.channel_files[i].c_str());
    }
    
    fclose(fp);
    printf("  ✓ Saved FX config to: %s\n", filename.c_str());
    return true;
}

// 保存频域多通道统计摘要（文本格式）- 不保存原始数据的版本
bool save_fx_multi_channel_summary_txt(const FxMultiChannelStats& stats,
                                       const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename.c_str());
        return false;
    }
    
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "            FX MULTI-CHANNEL CORRELATION SUMMARY (Frequency Domain)\n");
    fprintf(fp, "================================================================================\n\n");
    
    // 处理信息
    fprintf(fp, "1. PROCESSING INFORMATION\n");
    fprintf(fp, "------------------------\n");
    fprintf(fp, "  Timestamp            : %s\n", stats.timestamp_str.c_str());
    fprintf(fp, "  Number of channels   : %d\n", stats.config.num_channels);
    fprintf(fp, "  Total pairs analyzed : %d (Auto: %d, Cross: %d)\n", 
            (int)stats.pairs.size(), stats.config.num_channels, 
            stats.config.num_channels * (stats.config.num_channels - 1) / 2);
    fprintf(fp, "  FFT points per frame : %d\n", stats.n_per_frame);
    fprintf(fp, "  Number of frames     : %d\n", stats.num_frames);
    fprintf(fp, "  Total samples        : %zu (%.1f GB per channel)\n", 
            stats.total_samples,
            stats.total_samples * sizeof(complex_t) / 1024.0 / 1024.0 / 1024.0);
    fprintf(fp, "  Processing time      : %.2f ms\n", stats.processing_time_ms);
    fprintf(fp, "  Processing rate      : %.2f MS/s\n", stats.processing_rate_msps);
    fprintf(fp, "  Raw data saving      : Disabled (for performance)\n\n");
    
    // 通道配置
    fprintf(fp, "2. CHANNEL CONFIGURATION\n");
    fprintf(fp, "------------------------\n");
    for (int i = 0; i < stats.config.num_channels; i++) {
        fprintf(fp, "  CH%d:\n", i+1);
        fprintf(fp, "      Name: %s\n", stats.config.channel_names[i].c_str());
        fprintf(fp, "      File: %s\n", stats.config.channel_files[i].c_str());
    }
    fprintf(fp, "\n");
    
    // 每对数据的统计
    fprintf(fp, "3. PAIR STATISTICS\n");
    fprintf(fp, "------------------\n");
    
    for (const auto& pair : stats.pairs) {
        fprintf(fp, "  %s [%s]:\n", pair.pair_name.c_str(),
                (pair.type == FX_AUTO_CORRELATION) ? "AUTO-CORRELATION" : "CROSS-CORRELATION");
        fprintf(fp, "      Channels          : CH%d", pair.channel_i + 1);
        if (pair.type == FX_CROSS_CORRELATION) {
            fprintf(fp, " x CH%d\n", pair.channel_j + 1);
        } else {
            fprintf(fp, " (self)\n");
        }
        
        fprintf(fp, "      Spectrum Statistics:\n");
        fprintf(fp, "          Total power      : %.6e\n", pair.total_power);
        fprintf(fp, "          Maximum magnitude: %.6e at index %d\n", 
                pair.max_magnitude, pair.max_index);
        fprintf(fp, "          Frequency at max : %.2f MHz\n", 
                pair.max_freq_hz / 1e6);
        fprintf(fp, "          Average magnitude: %.6e\n", 
                pair.total_power / pair.spectrum_size);
        fprintf(fp, "          Frequency points : %d\n\n", pair.spectrum_size);
    }
    
    // 输出文件列表 - 修改后只列出实际生成的文件
    fprintf(fp, "4. OUTPUT FILES\n");
    fprintf(fp, "------------------\n");
    
    std::string prefix = filename.substr(0, filename.find_last_of("_"));
    prefix = prefix.substr(0, prefix.find_last_of("_"));
    
    fprintf(fp, "\n  4.1 Configuration Files:\n");
    fprintf(fp, "      %s_config.csv\n", prefix.c_str());
    fprintf(fp, "          Description: Channel configuration information\n");
    fprintf(fp, "          Format: CSV\n");
    fprintf(fp, "          Contents: Channel index, channel name, file path\n\n");
    
    fprintf(fp, "  4.2 Summary Files:\n");
    fprintf(fp, "      %s_summary.txt\n", prefix.c_str());
    fprintf(fp, "          Description: This human-readable summary file\n");
    fprintf(fp, "          Format: Text\n");
    fprintf(fp, "          Contents: Complete processing information and statistics\n\n");
    
    fprintf(fp, "  4.3 Correlation Spectra:\n");
    for (const auto& pair : stats.pairs) {
        fprintf(fp, "      %s_%s_spectrum.bin\n", 
                prefix.c_str(), pair.pair_name.c_str());
        fprintf(fp, "          Description: Complex cross-power spectrum for %s\n",
                pair.pair_name.c_str());
        fprintf(fp, "          Format: Binary (complex_t array)\n");
        fprintf(fp, "          Size: %d frequency points (%.2f MB)\n", 
                pair.spectrum_size,
                pair.spectrum_size * sizeof(complex_t) / 1024.0 / 1024.0);
        
        fprintf(fp, "      %s_%s_spectrum.csv\n", 
                prefix.c_str(), pair.pair_name.c_str());
        fprintf(fp, "          Description: Complex spectrum in CSV format\n");
        fprintf(fp, "          Format: CSV\n");
        fprintf(fp, "          Columns: frequency_index, frequency_hz, real_part, imag_part, magnitude, phase_deg\n\n");
    }
    
    // 文件统计
    fprintf(fp, "5. FILE STATISTICS\n");
    fprintf(fp, "------------------\n");
    
    int total_files = 1 +  // config.csv
                      1 +  // summary.txt
                      2 * stats.pairs.size();  // spectrum.bin + spectrum.csv per pair
    
    fprintf(fp, "  Total files generated : %d\n", total_files);
    fprintf(fp, "  Configuration files   : 1\n");
    fprintf(fp, "  Summary files         : 1\n");
    fprintf(fp, "  Spectrum files        : %d (%d pairs × 2 formats)\n\n", 
            2 * (int)stats.pairs.size(), (int)stats.pairs.size());
    
    fprintf(fp, "  Note: Raw signal data files are NOT saved (disabled for performance)\n\n");
    
    // 使用建议
    fprintf(fp, "6. USAGE NOTES\n");
    fprintf(fp, "------------------\n");
    fprintf(fp, "  - Spectrum files contain the accumulated cross-power spectrum\n");
    fprintf(fp, "  - Binary files are suitable for fast loading in MATLAB/Python\n");
    fprintf(fp, "  - CSV files can be opened in Excel or other spreadsheet software\n");
    fprintf(fp, "  - The spectrum shows power distribution across frequencies\n");
    fprintf(fp, "  - Frequency resolution = sample_rate / FFT_size = %.2f Hz\n", 
            stats.sample_rate_hz / stats.n_per_frame);
    
    fprintf(fp, "\n================================================================================\n");
    
    fclose(fp);
    printf("  ✓ Saved FX summary to: %s\n", filename.c_str());
    return true;
}

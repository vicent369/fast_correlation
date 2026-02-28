#ifndef FX_CORRELATOR_H
#define FX_CORRELATOR_H

#include <string>
#include <vector>
#include <ctime>
#include <cuda_runtime.h>  // 添加CUDA运行时头文件

#ifdef __cplusplus
extern "C" {
#endif

// 复数类型（与现有代码兼容）
typedef struct { 
    float x;  // 实部/I分量
    float y;  // 虚部/Q分量
} complex_t;

// ============ 多通道相关结果结构 ============

// 相关类型枚举
enum FxCorrelationType {
    FX_AUTO_CORRELATION,  // 自相关（频域）
    FX_CROSS_CORRELATION  // 互相关（频域）
};

// 频域相关对结果
struct FxCorrelationPairResult {
    FxCorrelationType type;
    int channel_i;                    // 通道i索引
    int channel_j;                    // 通道j索引
    std::string pair_name;             // 如 "CH1_AUTO", "CH1xCH2"
    
    // 累积的互功率谱（复数）
    std::vector<complex_t> accumulated_spectrum;
    int spectrum_size;                 // 频率点数
    
    // 统计信息
    float total_power;                  // 总功率
    float max_magnitude;                 // 最大幅度
    int max_index;                       // 最大幅度索引
    float max_freq_hz;                   // 最大幅度对应的频率
    
    // 每个频率点的幅度（用于统计）
    std::vector<float> magnitudes;
};

// 频域多通道配置
struct FxMultiChannelConfig {
    int num_channels;                    // 通道数 (1-8)
    std::vector<std::string> channel_files; // 每个通道的文件名
    std::vector<std::string> channel_names; // 通道名称
    
    FxMultiChannelConfig() : num_channels(0) {}
    
    // 获取总的相关对数量
    int get_total_pairs() const {
        if (num_channels <= 0) return 0;
        return num_channels + num_channels * (num_channels - 1) / 2;
    }
};

// 频域多通道统计
struct FxMultiChannelStats {
    FxMultiChannelConfig config;
    std::vector<FxCorrelationPairResult> pairs;
    
    // 处理参数
    int n_per_frame;                      // FFT点数
    int num_frames;                        // 累积帧数
    size_t total_samples;                   // 总样本数
    
    // 性能信息
    float processing_time_ms;
    float processing_rate_msps;
    float sample_rate_hz;
    
    std::string timestamp_str;
};

// ============ 原有函数声明 ============

/**
 * FX相关器 - 计算复数互功率谱
 * @param h_sig1      输入信号1（复数数组）
 * @param h_sig2      输入信号2（复数数组）
 * @param n_per_frame 每帧样本数（FFT点数）
 * @param num_frames  累积帧数
 * @param h_corr_out  输出：累积的互功率谱（复数，n_per_frame个点）
 * @param normalize   是否归一化（除以FFT点数）
 */
void gpu_fx_correlate(
    complex_t* h_sig1,
    complex_t* h_sig2,
    int n_per_frame,
    int num_frames,
    complex_t* h_corr_out,
    int normalize
);

/**
 * 自相关函数 - 计算复数自功率谱
 * @param h_signal    输入信号（复数数组）
 * @param n_per_frame 每帧样本数（FFT点数）
 * @param num_frames  累积帧数
 * @param h_auto_out  输出：累积的自功率谱（复数，n_per_frame个点）
 * @param normalize   是否归一化（除以FFT点数）
 */
void gpu_fx_auto_correlate(
    complex_t* h_signal,
    int n_per_frame,
    int num_frames,
    complex_t* h_auto_out,
    int normalize
);

// ============ Stream版本函数声明 ============

/**
 * FX相关器 - Stream版本（支持多个流并行）
 * @param h_sig1      输入信号1
 * @param h_sig2      输入信号2
 * @param n_per_frame 每帧样本数
 * @param num_frames  累积帧数
 * @param h_corr_out  输出谱
 * @param normalize   是否归一化
 * @param stream      CUDA流
 */
void gpu_fx_correlate_stream(
    complex_t* h_sig1,
    complex_t* h_sig2,
    int n_per_frame,
    int num_frames,
    complex_t* h_corr_out,
    int normalize,
    cudaStream_t stream
);

/**
 * 自相关函数 - Stream版本
 * @param h_signal    输入信号
 * @param n_per_frame 每帧样本数
 * @param num_frames  累积帧数
 * @param h_auto_out  输出谱
 * @param normalize   是否归一化
 * @param stream      CUDA流
 */
void gpu_fx_auto_correlate_stream(
    complex_t* h_signal,
    int n_per_frame,
    int num_frames,
    complex_t* h_auto_out,
    int normalize,
    cudaStream_t stream
);

/**
 * 批量处理所有相关对（使用多个流并行）
 * @param h_channel_data 所有通道的数据
 * @param all_pairs      所有相关对
 * @param n_per_frame    每帧样本数
 * @param num_frames     累积帧数
 * @param normalize      是否归一化
 */
void gpu_fx_correlate_batch(
    std::vector<complex_t*>& h_channel_data,
    std::vector<FxCorrelationPairResult>& all_pairs,
    int n_per_frame,
    int num_frames,
    int normalize
);

#ifdef __cplusplus
}
#endif

#endif

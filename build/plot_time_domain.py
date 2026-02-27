#!/usr/bin/env python3
"""
绘制C++时域互相关详细分析图（6子图布局）
完整修复版
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import csv
from datetime import datetime
import argparse

def read_csv_results(filename):
    """读取CSV结果文件"""
    results = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted_row = {}
            for key, value in row.items():
                if key in ['chunk_id', 'samples', 'peak_index', 'max_lag']:
                    converted_row[key] = int(value)
                elif key in ['peak_value', 'peak_real', 'peak_imag', 
                           'time_delay_ns', 'peak_power', 'timestamp']:
                    try:
                        converted_row[key] = float(value)
                    except ValueError:
                        converted_row[key] = value
                else:
                    converted_row[key] = value
            results.append(converted_row)
    return results

def load_time_domain_results(result_dir=None, results_file=None):
    """加载C++时域程序输出的结果文件"""
    if results_file:
        return read_csv_results(results_file)
    
    if result_dir is None:
        result_dir = 'time_domain_full'
    
    if not os.path.exists(result_dir):
        print(f"错误: 目录不存在 {result_dir}")
        return None
    
    result_files = glob.glob(os.path.join(result_dir, '*_results.csv'))
    if not result_files:
        print(f"错误: 在 {result_dir} 中找不到结果文件")
        return None
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"加载结果文件: {latest_file}")
    return read_csv_results(latest_file)

def load_correlation_data(results, result_dir):
    """加载互相关数据（适配新的多通道文件名格式）"""
    if not results:
        return None
    
    # 查找结果文件的基名
    result_file = None
    for file in glob.glob(os.path.join(result_dir, '*_results.csv')):
        if os.path.getctime(file) == max([os.path.getctime(f) for f in glob.glob(os.path.join(result_dir, '*_results.csv'))]):
            result_file = file
            break
    
    if result_file:
        base_name = os.path.basename(result_file).replace('_results.csv', '')
        
        # ===== 修改开始 =====
        # 查找互相关文件（CH1xCH2格式）
        corr_files = glob.glob(os.path.join(result_dir, f'{base_name}_CH?xCH?_first_chunk_corr.bin'))
        if corr_files:
            corr_file = corr_files[0]  # 取第一个互相关文件
            correlation_data = np.fromfile(corr_file, dtype=np.float32)
            print(f"加载互相关数据: {corr_file} ({len(correlation_data)} 个样本)")
            return correlation_data
        
        # 尝试CSV格式
        corr_csv_files = glob.glob(os.path.join(result_dir, f'{base_name}_CH?xCH?_first_chunk_corr.csv'))
        if corr_csv_files:
            corr_file = corr_csv_files[0]
            correlation_data = []
            with open(corr_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    correlation_data.append(float(row['correlation_value']))
            correlation_data = np.array(correlation_data)
            print(f"加载互相关数据: {corr_file} ({len(correlation_data)} 个样本)")
            return correlation_data
        
        # 如果找不到CH1xCH2，尝试其他模式
        print(f"在 {result_dir} 中找不到互相关文件")
        print(f"查找模式: {base_name}_CH?xCH?_first_chunk_corr.bin")
        # ===== 修改结束 =====
    
    return None

def load_saved_signal_data(result_dir, base_name=None):
    """从C++保存的文件中加载信号数据（适配新的文件名格式）"""
    if base_name is None:
        # 查找最新的信号文件
        sig_files = glob.glob(os.path.join(result_dir, '*_CH?_first_chunk.bin'))
        if not sig_files:
            print(f"在 {result_dir} 中找不到信号文件")
            return None, None
        
        # 从第一个信号文件提取base_name
        latest_file = max(sig_files, key=os.path.getctime)
        base_name = os.path.basename(latest_file).replace('_CH1_first_chunk.bin', '').replace('_CH2_first_chunk.bin', '')
    
    # 构建文件名（适配新格式）
    sig1_bin = os.path.join(result_dir, f'{base_name}_CH1_first_chunk.bin')
    sig2_bin = os.path.join(result_dir, f'{base_name}_CH2_first_chunk.bin')
    
    print(f"尝试加载信号数据: {sig1_bin} 和 {sig2_bin}")
    
    if os.path.exists(sig1_bin) and os.path.exists(sig2_bin):
        try:
            # 读取二进制文件
            sig1_raw = np.fromfile(sig1_bin, dtype=np.float32)
            sig2_raw = np.fromfile(sig2_bin, dtype=np.float32)
            
            print(f"  读取到: 信号1={len(sig1_raw)}字节, 信号2={len(sig2_raw)}字节")
            
            # 检查数据大小
            if len(sig1_raw) % 2 != 0 or len(sig2_raw) % 2 != 0:
                print(f"警告: 信号数据长度不是偶数 ({len(sig1_raw)}, {len(sig2_raw)})")
                return None, None
            
            # 转换为复数（交错存储：real, imag, real, imag, ...）
            sig1_complex = sig1_raw[::2] + 1j * sig1_raw[1::2]
            sig2_complex = sig2_raw[::2] + 1j * sig2_raw[1::2]
            
            print(f"  成功加载: 信号1={len(sig1_complex)}点, 信号2={len(sig2_complex)}点")
            return sig1_complex, sig2_complex
            
        except Exception as e:
            print(f"加载信号数据错误: {e}")
            return None, None
    
    print(f"信号文件不存在")
    return None, None

def generate_simulated_signal_data(num_samples, fs=100e6):
    """生成模拟信号数据用于演示"""
    t = np.arange(num_samples) / fs
    freq1 = 10e6  # 10MHz
    freq2 = 10.1e6  # 10.1MHz
    delay_samples = 5  # 5个样本延迟
    
    # 信号1: 正弦波 + 噪声
    sig1_i = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.1 * np.random.randn(num_samples)
    sig1_q = 0.5 * np.cos(2 * np.pi * freq1 * t) + 0.1 * np.random.randn(num_samples)
    sig1 = sig1_i + 1j * sig1_q
    
    # 信号2: 相同频率但有延迟和噪声
    sig2_i = 0.45 * np.sin(2 * np.pi * freq2 * (t - delay_samples/fs)) + 0.15 * np.random.randn(num_samples)
    sig2_q = 0.45 * np.cos(2 * np.pi * freq2 * (t - delay_samples/fs)) + 0.15 * np.random.randn(num_samples)
    sig2 = sig2_i + 1j * sig2_q
    
    return sig1, sig2

def fit_phase_frequency_curve(freq, phase, magnitude1=None, magnitude2=None):
    """对相位-频率曲线进行线性拟合"""
    if len(freq) < 10 or len(phase) < 10:
        return None, None, None, None, None
    
    # 解缠绕相位
    unwrapped_phase = np.unwrap(phase)
    
    # 线性拟合
    coeff = np.polyfit(freq, unwrapped_phase, 1)
    slope, intercept = coeff
    
    # 计算拟合值
    fit_phase = slope * freq + intercept
    
    # 计算R²
    residuals = unwrapped_phase - fit_phase
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((unwrapped_phase - np.mean(unwrapped_phase))**2)
    
    if SS_tot > 0:
        R2 = 1 - (SS_res / SS_tot)
    else:
        R2 = 0
    
    # 计算光程差和时延
    c = 3e8  # 光速 m/s
    delta_L = (slope * c) / (2 * np.pi)
    time_delay = delta_L / c
    
    return slope, intercept, R2, delta_L, time_delay

def create_detailed_analysis_6plots(correlation_data, results, sig1=None, sig2=None, 
                                   fs=100e6, file1='a0.dat', file2='b0.dat',
                                   output_dir='plots_time_domain'):
    """
    创建详细分析图（6子图布局）
    模仿dat_correlation_stream.py的create_detailed_analysis函数
    """
    if correlation_data is None:
        print("错误: 没有互相关数据")
        return
    
    n = len(correlation_data)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base1 = os.path.splitext(os.path.basename(file1))[0]
    base2 = os.path.splitext(os.path.basename(file2))[0]
    output_name = f"{output_dir}/detailed_analysis_6plots_{base1}_vs_{base2}_{timestamp}.png"
    
    # 创建图形 - 3行2列
    fig = plt.figure(figsize=(16, 12))
    
    # ===== 计算基本数据 =====
    # 计算延迟轴
    lags = np.arange(n) - (n - 1)
    time_delays_ns = lags / fs * 1e9
    
    # 找峰值
    peak_idx = np.argmax(correlation_data)
    peak_value = correlation_data[peak_idx]
    peak_delay_ns = time_delays_ns[peak_idx]
    max_lag = lags[peak_idx]
    
    # 归一化相关系数
    if sig1 is not None and sig2 is not None:
        norm_factor = np.sqrt(np.sum(np.abs(sig1)**2) * np.sum(np.abs(sig2)**2))
        normalized_corr = np.abs(peak_value) / norm_factor if norm_factor > 0 else 0
    else:
        normalized_corr = 0
    
    # ===== 第1行 =====
    # 图1: 互相关函数（左上）
    ax1 = plt.subplot(3, 2, 1)
    center_idx = n // 2
    view_range = min(500, n//2)
    
    ax1.plot(lags[center_idx-view_range:center_idx+view_range], 
             correlation_data[center_idx-view_range:center_idx+view_range], 
             'b-', linewidth=1)
    
    # 标记峰值
    peak_in_view = peak_idx - (center_idx - view_range)
    if 0 <= peak_in_view < 2*view_range:
        ax1.axvline(x=max_lag, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.plot(max_lag, peak_value, 'ro', markersize=6)
    
    ax1.set_title(f'Cross-Correlation Function\nLag={max_lag}, Delay={peak_delay_ns:.1f}ns')
    ax1.set_xlabel('Lag (samples)')
    ax1.set_ylabel('Correlation')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # 图2: 统计信息面板（右上）
    ax2 = plt.subplot(3, 2, 2)
    ax2.axis('off')
    
    # 从流式分析中获取统计信息
    if results and len(results) > 0:
        # 计算时延统计
        time_delays_all = [r['time_delay_ns'] for r in results if 'time_delay_ns' in r]
        peak_values_all = [r['peak_value'] for r in results if 'peak_value' in r]
        
        if time_delays_all:
            avg_time_delay = np.mean(time_delays_all)
            std_time_delay = np.std(time_delays_all)
        else:
            avg_time_delay = 0
            std_time_delay = 0
        
        if peak_values_all:
            avg_peak_value = np.mean(peak_values_all)
            std_peak_value = np.std(peak_values_all)
        else:
            avg_peak_value = 0
            std_peak_value = 0
    else:
        avg_time_delay = 0
        std_time_delay = 0
        avg_peak_value = 0
        std_peak_value = 0
    
    # 信号统计（如果提供了信号数据）
    if sig1 is not None and sig2 is not None:
        sig1_stats = f"μ={np.mean(sig1.real):.4f}, σ={np.std(sig1.real):.4f}"
        sig2_stats = f"μ={np.mean(sig2.real):.4f}, σ={np.std(sig2.real):.4f}"
    else:
        sig1_stats = "N/A"
        sig2_stats = "N/A"
    
    info_text = f"""DETAILED ANALYSIS - FIRST CHUNK
Files:
  {os.path.basename(file1)}
  {os.path.basename(file2)}

Current Chunk:
  Samples: {n:,}
  Duration: {n/fs*1000:.3f} ms
  Sample rate: {fs/1e6:.1f} MS/s

Correlation:
  Time delay: {peak_delay_ns:.1f} ns
  Lag: {max_lag} samples
  Peak value: {peak_value:.2e}
  Normalized corr: {normalized_corr:.4f}

Streaming Statistics:
  Avg delay: {avg_time_delay:.1f} ns
  Std delay: {std_time_delay:.1f} ns
  Avg peak: {avg_peak_value:.2e}
  Std peak: {std_peak_value:.2e}

Signal Statistics:
  Sig1: {sig1_stats}
  Sig2: {sig2_stats}
"""
    
    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # ===== 第2行 =====
    # 图3: I分量时域对比（左中）
    ax3 = plt.subplot(3, 2, 3)
    
    if sig1 is not None and sig2 is not None:
        plot_samples = min(200, len(sig1), len(sig2))
        sample_indices = np.arange(plot_samples)
        
        ax3.plot(sample_indices, np.real(sig1[:plot_samples]), 'b-', 
                label='Sig1 (I)', alpha=0.7, linewidth=1)
        ax3.plot(sample_indices, np.real(sig2[:plot_samples]), 'r-', 
                label='Sig2 (I)', alpha=0.7, linewidth=1)
        
        ax3.set_title('Time Domain - I Component')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Amplitude')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Signal data not available\n(Need to implement DAT file reading)', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Time Domain - I Component')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Amplitude')
        ax3.grid(True, alpha=0.3)
    
    # 图4: Q分量时域对比（右中）
    ax4 = plt.subplot(3, 2, 4)
    
    if sig1 is not None and sig2 is not None and np.iscomplexobj(sig1) and np.iscomplexobj(sig2):
        plot_samples = min(200, len(sig1), len(sig2))
        sample_indices = np.arange(plot_samples)
        
        ax4.plot(sample_indices, np.imag(sig1[:plot_samples]), 'b-', 
                label='Sig1 (Q)', alpha=0.7, linewidth=1)
        ax4.plot(sample_indices, np.imag(sig2[:plot_samples]), 'r-', 
                label='Sig2 (Q)', alpha=0.7, linewidth=1)
        
        ax4.set_title('Time Domain - Q Component')
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Amplitude')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Complex signal data not available\nor not IQ format', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Time Domain - Q Component')
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Amplitude')
        ax4.grid(True, alpha=0.3)
    
    # ===== 第3行 =====
    # 图5: 功率谱对比（左下）
    ax5 = plt.subplot(3, 2, 5)
    
    if sig1 is not None and sig2 is not None:
        N_fft = min(4096, len(sig1), len(sig2))
        if N_fft >= 256:
            fft1 = np.fft.fft(sig1[:N_fft])
            fft2 = np.fft.fft(sig2[:N_fft])
            freq = np.fft.fftfreq(N_fft, d=1/fs)
            
            # 只显示正频率部分
            positive_mask = freq >= 0
            positive_freq = freq[positive_mask]
            
            # 计算功率谱（dB）
            psd1 = 20 * np.log10(np.abs(fft1[positive_mask]) + 1e-10)
            psd2 = 20 * np.log10(np.abs(fft2[positive_mask]) + 1e-10)
            
            ax5.plot(positive_freq/1e6, psd1, 'b-', label='Sig1', alpha=0.7, linewidth=1)
            ax5.plot(positive_freq/1e6, psd2, 'r-', label='Sig2', alpha=0.7, linewidth=1)
            
            ax5.set_title('Power Spectra Comparison')
            ax5.set_xlabel('Frequency (MHz)')
            ax5.set_ylabel('Magnitude (dB)')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Insufficient data\nfor spectral analysis', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Power Spectra Comparison')
            ax5.set_xlabel('Frequency (MHz)')
            ax5.set_ylabel('Magnitude (dB)')
            ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Signal data not available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Power Spectra Comparison')
        ax5.set_xlabel('Frequency (MHz)')
        ax5.set_ylabel('Magnitude (dB)')
        ax5.grid(True, alpha=0.3)
    
    # 图6: 相位-频率关系图（右下）
    ax6 = plt.subplot(3, 2, 6)
    
    if sig1 is not None and sig2 is not None:
        N_fft = min(4096, len(sig1), len(sig2))
        if N_fft >= 256:
            fft1 = np.fft.fft(sig1[:N_fft])
            fft2 = np.fft.fft(sig2[:N_fft])
            freq = np.fft.fftfreq(N_fft, d=1/fs)
            
            # 计算互功率谱和相位差
            cross_power = fft1 * np.conj(fft2)
            phase_diff = np.angle(cross_power)
            
            # 只取正频率部分
            positive_mask = freq >= 0
            positive_phase = phase_diff[positive_mask]
            positive_freq = freq[positive_mask]
            
            # 绘制原始相位-频率图
            ax6.plot(positive_freq/1e6, np.degrees(positive_phase), 
                    'b-', linewidth=1, alpha=0.6, label='Phase difference')
            
            # 线性拟合
            slope, intercept, R2, delta_L, time_delay_fit = fit_phase_frequency_curve(
                positive_freq, positive_phase,
                np.abs(fft1[positive_mask]),
                np.abs(fft2[positive_mask])
            )
            
            if slope is not None:
                # 绘制拟合直线
                fit_phase = slope * positive_freq + intercept
                ax6.plot(positive_freq/1e6, np.degrees(fit_phase), 
                        'r-', linewidth=2, alpha=0.8, label='Linear fit')
                
                # 显示拟合结果
                result_text = f'Slope = {slope:.3e} rad/Hz\n'
                result_text += f'ΔL = {delta_L:.3f} m\n'
                result_text += f'τ = {time_delay_fit*1e9:.1f} ns\n'
                result_text += f'R² = {R2:.4f}'
                
                ax6.text(0.98, 0.98, result_text,
                        transform=ax6.transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # 显示拟合公式
                fit_eqn = f'Δφ(f) = {slope:.3e}·f + {intercept:.3f}'
                ax6.text(0.5, 0.05, fit_eqn,
                        transform=ax6.transAxes, fontsize=10,
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax6.set_xlim([0, positive_freq[-1]/1e6])
            ax6.set_title('Phase-Frequency Relationship')
            ax6.set_xlabel('Frequency (MHz)')
            ax6.set_ylabel('Phase Difference (degrees)')
            ax6.grid(True, alpha=0.3)
            ax6.legend(loc='best', fontsize=8)
        else:
            ax6.text(0.5, 0.5, 'Insufficient data\nfor phase analysis', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Phase-Frequency Relationship')
            ax6.set_xlabel('Frequency (MHz)')
            ax6.set_ylabel('Phase Difference (degrees)')
            ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Signal data not available', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Phase-Frequency Relationship')
        ax6.set_xlabel('Frequency (MHz)')
        ax6.set_ylabel('Phase Difference (degrees)')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"详细分析图（6子图）已保存: {output_name}")
    
    plt.show()
    
    return output_name

def create_time_series_plot(results, output_dir='plots_time_domain'):
    """创建时间序列摘要图"""
    if not results:
        return
    
    # 提取数据列
    chunk_ids = [r['chunk_id'] for r in results if 'chunk_id' in r]
    time_delays = [r['time_delay_ns'] for r in results if 'time_delay_ns' in r]
    peak_values = [r['peak_value'] for r in results if 'peak_value' in r]
    
    if not chunk_ids or not time_delays or not peak_values:
        return
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 时延随时间变化
    axes[0, 0].plot(chunk_ids, time_delays, 'b-', alpha=0.7)
    axes[0, 0].set_title('Time Delay Over Chunks')
    axes[0, 0].set_xlabel('Chunk ID')
    axes[0, 0].set_ylabel('Time Delay (ns)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 峰值幅度随时间变化
    axes[0, 1].plot(chunk_ids, peak_values, 'g-', alpha=0.7)
    axes[0, 1].set_title('Peak Correlation Over Chunks')
    axes[0, 1].set_xlabel('Chunk ID')
    axes[0, 1].set_ylabel('Peak Value')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # 时延直方图
    axes[1, 0].hist(time_delays, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_title('Time Delay Distribution')
    axes[1, 0].set_xlabel('Time Delay (ns)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 峰值直方图
    axes[1, 1].hist(peak_values, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title('Peak Value Distribution')
    axes[1, 1].set_xlabel('Peak Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{output_dir}/time_series_summary_{timestamp}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"时间序列摘要图已保存: {output_name}")
    
    plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制C++时域互相关详细分析图（6子图布局）')
    parser.add_argument('--input', '-i', help='输入结果文件 (.csv)')
    parser.add_argument('--dir', '-d', default='time_domain_full', 
                       help='结果目录 (默认: time_domain_full)')
    parser.add_argument('--output-dir', default='plots_time_domain', 
                       help='输出目录 (默认: plots_time_domain)')
    parser.add_argument('--fs', type=float, default=100e6,
                       help='采样率 (Hz) (默认: 100e6)')
    parser.add_argument('--signal-file1', help='第一个信号文件 (.dat)')
    parser.add_argument('--signal-file2', help='第二个信号文件 (.dat)')
    
    args = parser.parse_args()
    
    # 检查matplotlib是否可用
    try:
        import matplotlib
    except ImportError:
        print("错误: 需要安装matplotlib")
        print("请运行: sudo apt install python3-matplotlib")
        return
    
    print("=" * 70)
    print("DETAILED TIME DOMAIN CORRELATION ANALYSIS (6-PLOT LAYOUT)")
    print("=" * 70)
    
    # 加载结果数据
    results = load_time_domain_results(args.dir, args.input)
    
    if results is None:
        print("无法加载结果数据，退出...")
        return
    
    print(f"加载成功: {len(results)} 个数据块")
    
    # 加载互相关数据
    correlation_data = load_correlation_data(results, args.dir)
    
    if correlation_data is None:
        print("错误: 无法加载互相关数据")
        return
    
    print(f"互相关数据: {len(correlation_data)} 个样本")
    
    # 加载原始信号数据
    sig1, sig2 = None, None
    if args.signal_file1 and args.signal_file2:
        print("尝试从指定文件加载信号数据...")
        # 这里可以添加DAT文件读取功能
        # 暂时使用模拟数据
        sig1, sig2 = generate_simulated_signal_data(1024, args.fs)
        print("警告: DAT文件读取功能未实现，使用模拟数据")
    else:
        print("尝试从C++输出加载信号数据...")
        # 首先尝试从C++保存的文件加载
        sig1, sig2 = load_saved_signal_data(args.dir)
        
        # 如果加载失败，使用模拟数据
        if sig1 is None or sig2 is None:
            print("无法加载信号数据，使用模拟数据")
            sig1, sig2 = generate_simulated_signal_data(1024, args.fs)
    
    # 确定使用的文件名
    if results and 'file1' in results[0] and 'file2' in results[0]:
        file1 = results[0]['file1']
        file2 = results[0]['file2']
    else:
        file1 = args.signal_file1 if args.signal_file1 else 'a0.dat'
        file2 = args.signal_file2 if args.signal_file2 else 'b0.dat'
    
    # 创建详细分析图（6子图）
    create_detailed_analysis_6plots(correlation_data, results, sig1, sig2, 
                                   args.fs, file1, file2, args.output_dir)
    
    # 创建时间序列摘要图
    create_time_series_plot(results, args.output_dir)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print(f"图形保存到: {args.output_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    main()

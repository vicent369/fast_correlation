收集数据部分采用syncdaq程序，github地址为https://github.com/astrojhgu/syncdaq

数据分析部分
build文件夹里面有3个程序。
1.fast_correlation_td_full
时域相关函数分析。对于多路信号的数据流，可以分成若干个数据块（如1048576个样本点为一个数据块）。对于每个数据块，设置FFT点数（可以设为1024），计算时域相关函数。保存了每个数据块的峰值位置、时延、峰值功率，保存了各个数据块的峰值功率的累加结果；保存了多路信号的第一个数据块的完整的自相关和互相关函数（未对所有数据块的互相关函数积分），以及多路信号的第一个数据块的原始数据。
2.fast_correlation_fx
频域互功率谱分析。把2路信号的数据流分成若干帧，令每帧的样本数和FFT点数相同，对于2路信号的相同帧计算互功率谱（遍历帧内所有频率点），把所有帧的相同频率点的互功率谱累加（即对时间累积），保存每个频率处累加后的互功率谱及其实部虚部、幅度相位。
3.fast_correlation_fx_multi
和第二个程序基本一致，只不过支持多路信号的自相关和互相关计算。保存了多路信号的原始IQ数据，保存了多路信号的自相关谱和互相关谱。

使用方法
以3个通道a和b和c的数据为例，假设采集到的数据文件为a0.dat和b0.dat和c0.dat，存放在根目录下。
1.fast_correlation_td_full
进入build目录后，运行命令：
./fast_correlation_td_full -n ../a0.dat ../b0.dat ../c0.dat 1048576 1024 2
其中，-n为多通道模式（-2为双通道模式），1048576数据块大小，1024为FFT点数，2为运行模式（2为完整处理模式，1为测试模式，仅处理少量数据）
分析后的数据保存在build/multi_channel_results文件夹里。以summary结尾的txt文件总结了详细的输出文件，这里摘抄如下：
4. OUTPUT FILES
------------------

  4.1 Configuration Files:
      multi_channel_results/multi_20260227_config.csv
          Description: Channel configuration information
          Format: CSV
          Contents: Channel index, channel name, file path

  4.2 Result Files:
      multi_channel_results/multi_20260227_results.csv
          Description: Detailed chunk-by-chunk results for all pairs
          Format: CSV
          Contents: pair_name, type, chunk_id, peak_value, time_delay_ns, peak_power, peak_index

  4.3 Summary Files:
      multi_channel_results/multi_20260227_summary.csv
          Description: Statistical summary in CSV format
          Format: CSV
          Contents: Pair, Type, Mean Peak, Std Peak, Max Peak, Min Peak, Mean Delay, Std Delay, etc.

      multi_channel_results/multi_20260227_summary.txt
          Description: This human-readable summary file
          Format: Text
          Contents: Complete processing information and statistics

  4.4 Raw Signal Data (First Chunk):
      multi_channel_results/multi_20260227_CH1_first_chunk.bin
          Description: Raw signal data for channel CH1 (first chunk only)
          Format: Binary (float2_data structure)
          Size: 1024000 samples (7.8 MB)
      multi_channel_results/multi_20260227_CH1_first_chunk.csv
          Description: Raw signal data in CSV format
          Format: CSV
          Columns: index, time_ns, real_part, imag_part, magnitude, phase_deg

      multi_channel_results/multi_20260227_CH2_first_chunk.bin
          Description: Raw signal data for channel CH2 (first chunk only)
          Format: Binary (float2_data structure)
          Size: 1024000 samples (7.8 MB)
      multi_channel_results/multi_20260227_CH2_first_chunk.csv
          Description: Raw signal data in CSV format
          Format: CSV
          Columns: index, time_ns, real_part, imag_part, magnitude, phase_deg

  4.5 Correlation Data (First Chunk):
      multi_channel_results/multi_20260227_CH1_AUTO_first_chunk_corr.bin
          Description: Complete correlation function for CH1_AUTO (first chunk only)
          Format: Binary (float array)
          Size: 1024 points (0.0 MB)
      multi_channel_results/multi_20260227_CH1_AUTO_first_chunk_corr.csv
          Description: Complete correlation function in CSV format
          Format: CSV
          Columns: index, lag_samples, time_delay_ns, correlation_value

      multi_channel_results/multi_20260227_CH2_AUTO_first_chunk_corr.bin
          Description: Complete correlation function for CH2_AUTO (first chunk only)
          Format: Binary (float array)
          Size: 1024 points (0.0 MB)
      multi_channel_results/multi_20260227_CH2_AUTO_first_chunk_corr.csv
          Description: Complete correlation function in CSV format
          Format: CSV
          Columns: index, lag_samples, time_delay_ns, correlation_value

      multi_channel_results/multi_20260227_CH1xCH2_first_chunk_corr.bin
          Description: Complete correlation function for CH1xCH2 (first chunk only)
          Format: Binary (float array)
          Size: 1024 points (0.0 MB)
      multi_channel_results/multi_20260227_CH1xCH2_first_chunk_corr.csv
          Description: Complete correlation function in CSV format
          Format: CSV
          Columns: index, lag_samples, time_delay_ns, correlation_value
画图功能由build/plot_time_domain.py实现。进入build文件夹，运行命令：
python3 plot_time_domain.py --dir multi_channel_results/
画图结果存在build/plots_time_domain文件夹里。其中detailed_analysis_6plots开头的png文件为6子图详细分析（以第一个数据块的互相关函数和信号的原始数据为基础，画出互相关函数图、多路信号的时域IQ分量图、功率谱对比图、频率-相位关系图和光程差等统计信息），time_series_summary开头的png文件为时间序列摘要图（时延和相关峰值随数据块的变化图（即随时间的变化图）、时延和相关性分布的直方图）。
2.fast_correlation_fx
进入build目录后，运行命令：
./fast_correlation_fx ../a0.dat ../b0.dat 1024 40000 0
其中，第3个参数为FFT点数（即每帧样本数），第4个参数为帧数，第5个参数为起始样本偏移量（从文件开头算起，单位：样本数）
分析后的数据保存在build/fx_correlation_results文件夹里。以summary结尾的txt文件总结了详细的输出文件，这里摘抄如下：
complex_spectrum.csv - Complex spectrum (CSV)
complex_spectrum.bin - Complex spectrum (binary)
summary.txt - This summary
3.fast_correlation_fx_multi
进入build目录后，运行命令：
./fast_correlation_fx_multi -n ../a0.dat ../b0.dat ../c0.dat 65536 625
其中，-n代表多通道模式，65536为FFT点数（即每帧样本数），625为帧数
分析后的数据保存在build/fx_multi_results文件夹里。以summary结尾的txt文件总结了详细的输出文件，这里摘抄如下：
4. OUTPUT FILES
------------------

  4.1 Configuration Files:
      fx_multi_results/fx_multi_20260227_155725_config.csv
          Description: Channel configuration information
          Format: CSV

  4.2 Summary Files:
      fx_multi_results/fx_multi_20260227_155725_summary.txt
          Description: This human-readable summary file
          Format: Text

  4.3 Raw Signal Data:
      fx_multi_results/fx_multi_20260227_155725_CH1_raw.bin
          Description: Raw signal data for channel CH1
          Format: Binary (complex_t structure)
          Size: 40960000 samples (312.5 MB)

      fx_multi_results/fx_multi_20260227_155725_CH2_raw.bin
          Description: Raw signal data for channel CH2
          Format: Binary (complex_t structure)
          Size: 40960000 samples (312.5 MB)

  4.4 Correlation Spectra:
      fx_multi_results/fx_multi_20260227_155725_CH1_AUTO_spectrum.bin
          Description: Complex cross-power spectrum for CH1_AUTO
          Format: Binary (complex_t array)
          Size: 65536 frequency points (0.5 MB)
      fx_multi_results/fx_multi_20260227_155725_CH1_AUTO_spectrum.csv
          Description: Complex spectrum in CSV format
          Format: CSV
          Columns: frequency_index, frequency_hz, real_part, imag_part, magnitude, phase_deg

      fx_multi_results/fx_multi_20260227_155725_CH2_AUTO_spectrum.bin
          Description: Complex cross-power spectrum for CH2_AUTO
          Format: Binary (complex_t array)
          Size: 65536 frequency points (0.5 MB)
      fx_multi_results/fx_multi_20260227_155725_CH2_AUTO_spectrum.csv
          Description: Complex spectrum in CSV format
          Format: CSV
          Columns: frequency_index, frequency_hz, real_part, imag_part, magnitude, phase_deg

      fx_multi_results/fx_multi_20260227_155725_CH1xCH2_spectrum.bin
          Description: Complex cross-power spectrum for CH1xCH2
          Format: Binary (complex_t array)
          Size: 65536 frequency points (0.5 MB)
      fx_multi_results/fx_multi_20260227_155725_CH1xCH2_spectrum.csv
          Description: Complex spectrum in CSV format
          Format: CSV
          Columns: frequency_index, frequency_hz, real_part, imag_part, magnitude, phase_deg

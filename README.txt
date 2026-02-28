收集数据部分采用syncdaq程序，github地址为https://github.com/astrojhgu/syncdaq
把pcie的100G以太网卡插入主机。将100G光纤连接主机和T510采集板，并设置两端的光口IP地址。把千兆光纤（网口）连接采集板和交换机，用另一根千兆光纤连接主机和交换机，通过DHCP获取两个网口的IP地址，其中板卡的可以用picocom获取。
使用基于rust语言的“syncdaq”控制软件，在XGbeCfgSingle.yaml文件里配置好信号源（板卡）和接收端（主机）的MAC地址、IP地址（IP用光口的，相当于另建一个局域网）和端口后，调整100G网卡的MTU为9000，监听板卡网口端口，设置内部gps为时钟和pps源，执行mts同步，在MixerSet.yaml文件里修改本振频率，传输并接收数据。
具体地，抓取基带数据可以用如下命令：
cargo run --bin capture_pipeline --release -- -a '0.0.0.0:4000' -F a -k 10000
其中，0.0.0.0要修改为板卡的网口IP地址，4000要修改为主机的端口；-F后面的a代表数据文件名，-F后面跟"full dump file"，即记录所有的数据包，并根据-k后面的参数分段；-k后面的10000代表每个文件的数据包大小，如果不给-k参数，则不分段，所有数据都流入一个文件里；-o后面跟的是普通的dump file，每次满足要求就覆盖，这里的要求有：-n，每次dump时，dump的帧数；-m，每隔多少个包dump一次；-p，总共dump多少个以太网帧。

数据分析部分
build文件夹里面有2个程序。
1.fast_correlation_td_full
时域相关函数分析。对于多路信号的数据流，可以分成若干个数据块（如1048576个样本点为一个数据块）。对于每个数据块，设置FFT点数（可以设为1024），计算时域相关函数。保存了每个数据块的峰值位置、时延、峰值功率，保存了各个数据块的峰值功率的累加结果；保存了多路信号的第一个数据块的完整的自相关和互相关函数（未对所有数据块的互相关函数积分），以及多路信号的第一个数据块的原始数据。
2.fast_correlation_fx_multi
频域互功率谱分析。把多路信号的数据流分成若干帧，令每帧的样本数和FFT点数相同，支持多路信号的自相关和互相关计算，对于多路信号的相同帧计算互功率谱（遍历帧内所有频率点），把所有帧的相同频率点的互功率谱累加（即对时间累积），保存每个频率处累加后的互功率谱及其实部虚部、幅度相位，保存多路信号的自相关谱和互相关谱。自相关和互相关的计算采用CUDA stream。

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
      multi_channel_results/multi_20260228_config.csv
          Description: Channel configuration information
          Format: CSV
          Contents: Channel index, channel name, file path

  4.2 Result Files:
      multi_channel_results/multi_20260228_results.csv
          Description: Detailed chunk-by-chunk results for all pairs
          Format: CSV
          Contents: pair_name, type, chunk_id, peak_value, time_delay_ns, peak_power, peak_index

  4.3 Summary Files:
      multi_channel_results/multi_20260228_summary.csv
          Description: Statistical summary in CSV format
          Format: CSV
          Contents: Pair, Type, Mean Peak, Std Peak, Max Peak, Min Peak, Mean Delay, Std Delay, etc.

      multi_channel_results/multi_20260228_summary.txt
          Description: This human-readable summary file
          Format: Text
          Contents: Complete processing information and statistics

  4.4 Raw Signal Data (First Chunk):
      multi_channel_results/multi_20260228_CH1_first_chunk.bin
          Description: Raw signal data for channel CH1 (first chunk only)
          Format: Binary (float2_data structure)
          Size: 1044897 samples (8.0 MB)
      multi_channel_results/multi_20260228_CH1_first_chunk.csv
          Description: Raw signal data in CSV format
          Format: CSV
          Columns: index, time_ns, real_part, imag_part, magnitude, phase_deg

      multi_channel_results/multi_20260228_CH2_first_chunk.bin
          Description: Raw signal data for channel CH2 (first chunk only)
          Format: Binary (float2_data structure)
          Size: 1044897 samples (8.0 MB)
      multi_channel_results/multi_20260228_CH2_first_chunk.csv
          Description: Raw signal data in CSV format
          Format: CSV
          Columns: index, time_ns, real_part, imag_part, magnitude, phase_deg

      multi_channel_results/multi_20260228_CH3_first_chunk.bin
          Description: Raw signal data for channel CH3 (first chunk only)
          Format: Binary (float2_data structure)
          Size: 1044897 samples (8.0 MB)
      multi_channel_results/multi_20260228_CH3_first_chunk.csv
          Description: Raw signal data in CSV format
          Format: CSV
          Columns: index, time_ns, real_part, imag_part, magnitude, phase_deg

  4.5 Correlation Data (First Chunk):
      multi_channel_results/multi_20260228_CH1_AUTO_first_chunk_corr.bin
          Description: Complete correlation function for CH1_AUTO (first chunk only)
          Format: Binary (float array)
          Size: 1024 points (0.0 MB)
      multi_channel_results/multi_20260228_CH1_AUTO_first_chunk_corr.csv
          Description: Complete correlation function in CSV format
          Format: CSV
          Columns: index, lag_samples, time_delay_ns, correlation_value

      multi_channel_results/multi_20260228_CH2_AUTO_first_chunk_corr.bin
          Description: Complete correlation function for CH2_AUTO (first chunk only)
          Format: Binary (float array)
          Size: 1024 points (0.0 MB)
      multi_channel_results/multi_20260228_CH2_AUTO_first_chunk_corr.csv
          Description: Complete correlation function in CSV format
          Format: CSV
          Columns: index, lag_samples, time_delay_ns, correlation_value

      multi_channel_results/multi_20260228_CH3_AUTO_first_chunk_corr.bin
          Description: Complete correlation function for CH3_AUTO (first chunk only)
          Format: Binary (float array)
          Size: 1024 points (0.0 MB)
      multi_channel_results/multi_20260228_CH3_AUTO_first_chunk_corr.csv
          Description: Complete correlation function in CSV format
          Format: CSV
          Columns: index, lag_samples, time_delay_ns, correlation_value

      multi_channel_results/multi_20260228_CH1xCH2_first_chunk_corr.bin
          Description: Complete correlation function for CH1xCH2 (first chunk only)
          Format: Binary (float array)
          Size: 1024 points (0.0 MB)
      multi_channel_results/multi_20260228_CH1xCH2_first_chunk_corr.csv
          Description: Complete correlation function in CSV format
          Format: CSV
          Columns: index, lag_samples, time_delay_ns, correlation_value

      multi_channel_results/multi_20260228_CH1xCH3_first_chunk_corr.bin
          Description: Complete correlation function for CH1xCH3 (first chunk only)
          Format: Binary (float array)
          Size: 1024 points (0.0 MB)
      multi_channel_results/multi_20260228_CH1xCH3_first_chunk_corr.csv
          Description: Complete correlation function in CSV format
          Format: CSV
          Columns: index, lag_samples, time_delay_ns, correlation_value

      multi_channel_results/multi_20260228_CH2xCH3_first_chunk_corr.bin
          Description: Complete correlation function for CH2xCH3 (first chunk only)
          Format: Binary (float array)
          Size: 1024 points (0.0 MB)
      multi_channel_results/multi_20260228_CH2xCH3_first_chunk_corr.csv
          Description: Complete correlation function in CSV format
          Format: CSV
          Columns: index, lag_samples, time_delay_ns, correlation_value
画图功能由build/plot_time_domain.py实现。进入build文件夹，运行命令：
python3 plot_time_domain.py --dir multi_channel_results/
画图结果存在build/plots_time_domain文件夹里。其中detailed_analysis_6plots开头的png文件为6子图详细分析（以第一个数据块的互相关函数和信号的原始数据为基础，画出互相关函数图、多路信号的时域IQ分量图、功率谱对比图、频率-相位关系图和光程差等统计信息），time_series_summary开头的png文件为时间序列摘要图（时延和相关峰值随数据块的变化图（即随时间的变化图）、时延和相关性分布的直方图）。

2.fast_correlation_fx_multi
进入build目录后，运行命令：
./fast_correlation_fx_multi -n ../a0.dat ../b0.dat ../c0.dat 65536 625
其中，-n代表多通道模式，65536为FFT点数（即每帧样本数），625为帧数
分析后的数据保存在build/fx_multi_results文件夹里。以summary结尾的txt文件总结了详细的输出文件，这里摘抄如下：
4. OUTPUT FILES
------------------

  4.1 Configuration Files:
      fx_multi_results/fx_multi_20260228_config.csv
          Description: Channel configuration information
          Format: CSV
          Contents: Channel index, channel name, file path

  4.2 Summary Files:
      fx_multi_results/fx_multi_20260228_summary.txt
          Description: This human-readable summary file
          Format: Text
          Contents: Complete processing information and statistics

  4.3 Correlation Spectra:
      fx_multi_results/fx_multi_20260228_CH1_AUTO_spectrum.bin
          Description: Complex cross-power spectrum for CH1_AUTO
          Format: Binary (complex_t array)
          Size: 65536 frequency points (0.50 MB)
      fx_multi_results/fx_multi_20260228_CH1_AUTO_spectrum.csv
          Description: Complex spectrum in CSV format
          Format: CSV
          Columns: frequency_index, frequency_hz, real_part, imag_part, magnitude, phase_deg

      fx_multi_results/fx_multi_20260228_CH2_AUTO_spectrum.bin
          Description: Complex cross-power spectrum for CH2_AUTO
          Format: Binary (complex_t array)
          Size: 65536 frequency points (0.50 MB)
      fx_multi_results/fx_multi_20260228_CH2_AUTO_spectrum.csv
          Description: Complex spectrum in CSV format
          Format: CSV
          Columns: frequency_index, frequency_hz, real_part, imag_part, magnitude, phase_deg

      fx_multi_results/fx_multi_20260228_CH3_AUTO_spectrum.bin
          Description: Complex cross-power spectrum for CH3_AUTO
          Format: Binary (complex_t array)
          Size: 65536 frequency points (0.50 MB)
      fx_multi_results/fx_multi_20260228_CH3_AUTO_spectrum.csv
          Description: Complex spectrum in CSV format
          Format: CSV
          Columns: frequency_index, frequency_hz, real_part, imag_part, magnitude, phase_deg

      fx_multi_results/fx_multi_20260228_CH1xCH2_spectrum.bin
          Description: Complex cross-power spectrum for CH1xCH2
          Format: Binary (complex_t array)
          Size: 65536 frequency points (0.50 MB)
      fx_multi_results/fx_multi_20260228_CH1xCH2_spectrum.csv
          Description: Complex spectrum in CSV format
          Format: CSV
          Columns: frequency_index, frequency_hz, real_part, imag_part, magnitude, phase_deg

      fx_multi_results/fx_multi_20260228_CH1xCH3_spectrum.bin
          Description: Complex cross-power spectrum for CH1xCH3
          Format: Binary (complex_t array)
          Size: 65536 frequency points (0.50 MB)
      fx_multi_results/fx_multi_20260228_CH1xCH3_spectrum.csv
          Description: Complex spectrum in CSV format
          Format: CSV
          Columns: frequency_index, frequency_hz, real_part, imag_part, magnitude, phase_deg

      fx_multi_results/fx_multi_20260228_CH2xCH3_spectrum.bin
          Description: Complex cross-power spectrum for CH2xCH3
          Format: Binary (complex_t array)
          Size: 65536 frequency points (0.50 MB)
      fx_multi_results/fx_multi_20260228_CH2xCH3_spectrum.csv
          Description: Complex spectrum in CSV format
          Format: CSV
          Columns: frequency_index, frequency_hz, real_part, imag_part, magnitude, phase_deg

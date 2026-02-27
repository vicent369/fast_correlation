#ifndef DAT_READER_H
#define DAT_READER_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// 重命名为避免与CUDA的float2冲突
typedef struct { float x, y; } float2_data;

// 读取DAT文件（IQ int16格式）
float2_data* read_dat_file(const char* filename, size_t* num_samples);
// 读取部分数据（用于测试）
float2_data* read_dat_partial(const char* filename, size_t start_sample, size_t num_samples);
// 释放内存
void free_data(float2_data* data);

#ifdef __cplusplus
}
#endif

#endif

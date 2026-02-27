#include "dat_reader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

float2_data* read_dat_file(const char* filename, size_t* num_samples) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return nullptr;
    }
    
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    rewind(fp);
    
    // IQ int16: 每个样本4字节
    *num_samples = file_size / 4;
    
    printf("  File size: %zu bytes, Samples: %zu\n", file_size, *num_samples);
    
    // 读取原始数据
    short* raw_data = (short*)malloc(file_size);
    if (!raw_data) {
        printf("Error: Failed to allocate memory for raw data\n");
        fclose(fp);
        return nullptr;
    }
    
    size_t read_size = fread(raw_data, 1, file_size, fp);
    fclose(fp);
    
    if (read_size != file_size) {
        printf("Error: Failed to read entire file (read %zu of %zu bytes)\n", 
               read_size, file_size);
        free(raw_data);
        return nullptr;
    }
    
    // 转换为复数
    float2_data* complex_data = (float2_data*)malloc(*num_samples * sizeof(float2_data));
    if (!complex_data) {
        printf("Error: Failed to allocate memory for complex data\n");
        free(raw_data);
        return nullptr;
    }
    
    // 转换
    for (size_t i = 0; i < *num_samples; i++) {
        complex_data[i].x = (float)raw_data[2*i];      // I
        complex_data[i].y = (float)raw_data[2*i + 1];  // Q
    }
    
    free(raw_data);
    return complex_data;
}

float2_data* read_dat_partial(const char* filename, size_t start_sample, size_t num_samples) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return nullptr;
    }
    
    // 计算字节位置
    size_t start_byte = start_sample * 4;  // 每个样本4字节
    size_t bytes_to_read = num_samples * 4;
    
    // 定位
    if (fseek(fp, start_byte, SEEK_SET) != 0) {
        printf("Error: Failed to seek in file\n");
        fclose(fp);
        return nullptr;
    }
    
    // 分配内存
    short* raw_data = (short*)malloc(bytes_to_read);
    if (!raw_data) {
        printf("Error: Failed to allocate memory\n");
        fclose(fp);
        return nullptr;
    }
    
    // 读取
    size_t read_size = fread(raw_data, 1, bytes_to_read, fp);
    fclose(fp);
    
    if (read_size != bytes_to_read) {
        printf("Error: Partial read (read %zu of %zu bytes)\n", 
               read_size, bytes_to_read);
        free(raw_data);
        return nullptr;
    }
    
    // 转换
    float2_data* complex_data = (float2_data*)malloc(num_samples * sizeof(float2_data));
    if (!complex_data) {
        printf("Error: Failed to allocate complex memory\n");
        free(raw_data);
        return nullptr;
    }
    
    for (size_t i = 0; i < num_samples; i++) {
        complex_data[i].x = (float)raw_data[2*i];
        complex_data[i].y = (float)raw_data[2*i + 1];
    }
    
    free(raw_data);
    return complex_data;
}

void free_data(float2_data* data) {
    if (data) {
        free(data);
    }
}

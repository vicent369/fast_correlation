cmake_minimum_required(VERSION 3.18)
project(FastCorrelationFixed CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA设置
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "61")  # GTX 1080 Ti

# 优化设置
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -ffast-math)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# ============ 时域完整输出版本 ============
set(TD_FULL_SOURCES
    src/main_td.cpp
    src/time_domain_output.cpp
    src/correlation_td.cu
    src/dat_reader.cpp
)

set(BUILD_TD_FULL TRUE)
foreach(src_file ${TD_FULL_SOURCES})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src_file})
        message(WARNING "File ${src_file} not found. Time-domain full version will not be built.")
        set(BUILD_TD_FULL FALSE)
    endif()
endforeach()

if(BUILD_TD_FULL)
    add_executable(fast_correlation_td_full
        src/main_td.cpp
        src/time_domain_output.cpp
        src/correlation_td.cu
        src/dat_reader.cpp
    )
    
    target_include_directories(fast_correlation_td_full PRIVATE include)
    target_link_libraries(fast_correlation_td_full PRIVATE cufft)
    
    message(STATUS "Building time-domain full version: fast_correlation_td_full")
endif()

# ============ 新增：FX相关器多通道版本 ============
set(FX_MULTI_SOURCES
    src/main_fx_multi.cpp
    src/fx_correlator.cu
    src/fx_output.cpp
    src/dat_reader.cpp
)

set(BUILD_FX_MULTI TRUE)
foreach(src_file ${FX_MULTI_SOURCES})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src_file})
        message(WARNING "File ${src_file} not found. FX multi-channel correlator will not be built.")
        set(BUILD_FX_MULTI FALSE)
    endif()
endforeach()

if(BUILD_FX_MULTI)
    message(STATUS "Building FX multi-channel correlator: fast_correlation_fx_multi")
    
    add_executable(fast_correlation_fx_multi
        src/main_fx_multi.cpp
        src/fx_correlator.cu
        src/fx_output.cpp
        src/dat_reader.cpp
    )
    
    target_include_directories(fast_correlation_fx_multi PRIVATE include)
    target_link_libraries(fast_correlation_fx_multi PRIVATE cufft)
    
    # 为CUDA文件设置单独的语言
    set_source_files_properties(src/fx_correlator.cu PROPERTIES LANGUAGE CUDA)
    
    # 创建输出目录的构建后命令
    add_custom_command(TARGET fast_correlation_fx_multi POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "fx_multi_results"
        COMMENT "Creating output directory for FX multi-channel correlator results"
    )
endif()

# ============ 安装目标 ============
if(BUILD_TD_FULL)
    install(TARGETS fast_correlation_td_full DESTINATION bin)
endif()

if(BUILD_FX_MULTI)
    install(TARGETS fast_correlation_fx_multi DESTINATION bin)
endif()

# ============ 伪目标：构建所有 ============
add_custom_target(all_correlation
    COMMENT "Building all correlation analyzers"
)

if(BUILD_TD_FULL)
    add_dependencies(all_correlation fast_correlation_td_full)
endif()

if(BUILD_FX_MULTI)
    add_dependencies(all_correlation fast_correlation_fx_multi)
endif()

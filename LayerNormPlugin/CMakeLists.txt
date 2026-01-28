cmake_minimum_required(VERSION 3.10)
project(layernorm_plugin)

# 设置CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找TensorRT
find_package(TensorRT REQUIRED)
include_directories(${TENSORRT_INCLUDE_DIRS})

# 设置CUDA包含路径
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# 编译插件库
add_library(layernorm_plugin SHARED
    layernorm_plugin.cpp
    layernorm_kernel.cu
)

# 链接TensorRT
target_link_libraries(layernorm_plugin
    ${TENSORRT_LIBRARY}
    cudart
)

# 设置编译选项
target_compile_definitions(layernorm_plugin PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=1
)

# 设置CUDA编译选项
set_target_properties(layernorm_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
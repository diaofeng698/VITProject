#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

// 使用更稳定的 epsilon
constexpr float kLAYERNORM_EPSILON = 1e-6f;

// CUDA kernel: LayerNorm 前向计算 (FP32)
__global__ void layer_norm_kernel_float(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    float epsilon,
    int rows,
    int cols,
    int elems_per_row
) {
    extern __shared__ float s_data[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // 第一步: 计算均值
    float mean = 0.0f;
    for (int i = tid; i < elems_per_row; i += blockDim.x) {
        int idx = row * elems_per_row + i;
        if (i < cols) {
            mean += input[idx];
        }
    }
    
    // 共享内存归约计算均值
    __shared__ float s_mean;
    s_data[tid] = mean;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        s_mean = s_data[0] / cols;
    }
    __syncthreads();
    mean = s_mean;
    
    // 第二步: 计算方差
    float variance = 0.0f;
    for (int i = tid; i < elems_per_row; i += blockDim.x) {
        int idx = row * elems_per_row + i;
        if (i < cols) {
            float val = input[idx];
            float diff = val - mean;
            variance += diff * diff;
        }
    }
    
    // 共享内存归约计算方差
    s_data[tid] = variance;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        s_mean = s_data[0] / cols;
    }
    __syncthreads();
    variance = s_mean;
    
    // 第三步: 归一化并应用缩放和偏移
    float rstd = rsqrtf(variance + epsilon);
    
    for (int i = tid; i < elems_per_row; i += blockDim.x) {
        int idx = row * elems_per_row + i;
        if (i < cols) {
            float val = input[idx];
            float normalized = (val - mean) * rstd;
            output[idx] = normalized * gamma[i] + beta[i];
        }
    }
}

// CUDA kernel: LayerNorm 前向计算 (FP16)
__global__ void layer_norm_kernel_half(
    __half* output,
    const __half* input,
    const __half* gamma,
    const __half* beta,
    float epsilon,
    int rows,
    int cols,
    int elems_per_row
) {
    extern __shared__ float s_data[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // 计算均值
    float mean = 0.0f;
    for (int i = tid; i < elems_per_row; i += blockDim.x) {
        int idx = row * elems_per_row + i;
        if (i < cols) {
            mean += __half2float(input[idx]);
        }
    }
    
    __shared__ float s_mean;
    s_data[tid] = mean;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        s_mean = s_data[0] / cols;
    }
    __syncthreads();
    mean = s_mean;
    
    // 计算方差
    float variance = 0.0f;
    for (int i = tid; i < elems_per_row; i += blockDim.x) {
        int idx = row * elems_per_row + i;
        if (i < cols) {
            float val = __half2float(input[idx]);
            float diff = val - mean;
            variance += diff * diff;
        }
    }
    
    s_data[tid] = variance;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        s_mean = s_data[0] / cols;
    }
    __syncthreads();
    variance = s_mean;
    
    // 归一化
    float rstd = rsqrtf(variance + epsilon);
    
    for (int i = tid; i < elems_per_row; i += blockDim.x) {
        int idx = row * elems_per_row + i;
        if (i < cols) {
            float val = __half2float(input[idx]);
            float normalized = (val - mean) * rstd;
            float scale = __half2float(gamma[i]);
            float bias = __half2float(beta[i]);
            output[idx] = __float2half_rn(normalized * scale + bias);
        }
    }
}

// 包装函数 - FP32
extern "C" void launch_layer_norm_float(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    float epsilon,
    int rows,
    int cols,
    cudaStream_t stream
) {
    int elems_per_row = (cols + 127) / 128 * 128;
    int threads_per_block = 256;
    int blocks_per_grid = rows;
    int shared_mem_size = threads_per_block * sizeof(float);
    
    layer_norm_kernel_float<<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(
        output, input, gamma, beta, epsilon, rows, cols, elems_per_row
    );
}

// 包装函数 - FP16
extern "C" void launch_layer_norm_half(
    __half* output,
    const __half* input,
    const __half* gamma,
    const __half* beta,
    float epsilon,
    int rows,
    int cols,
    cudaStream_t stream
) {
    int elems_per_row = (cols + 127) / 128 * 128;
    int threads_per_block = 256;
    int blocks_per_grid = rows;
    int shared_mem_size = threads_per_block * sizeof(float);
    
    layer_norm_kernel_half<<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(
        output, input, gamma, beta, epsilon, rows, cols, elems_per_row
    );
}
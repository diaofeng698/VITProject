# Compiler settings
NVCC := /usr/local/cuda/bin/nvcc
CXX := g++

# CUDA settings
CUDA_PATH := /usr/local/cuda
CUDA_INCLUDE := $(CUDA_PATH)/include
CUDA_LIB := $(CUDA_PATH)/lib64

# TensorRT settings (adjust path if needed)
TENSORRT_PATH := /usr/local/TensorRT
TENSORRT_INCLUDE := $(TENSORRT_PATH)/include
TENSORRT_LIB := $(TENSORRT_PATH)/lib

# Compiler flags
CXXFLAGS := -std=c++14 -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=1
NVCCFLAGS := -std=c++14 -Xcompiler -fPIC -O3

# Include paths
INCLUDES := -I$(CUDA_INCLUDE) -I$(TENSORRT_INCLUDE)

# Library paths and libraries
LDFLAGS := -L$(CUDA_LIB) -L$(TENSORRT_LIB)
LIBS := -lnvinfer -lcudart

# Target
TARGET := liblayernorm_plugin.so

# Source files
CPP_SRCS := layernorm_plugin.cpp
CU_SRCS := layernorm_kernel.cu

# Object files
CPP_OBJS := $(CPP_SRCS:.cpp=.o)
CU_OBJS := $(CU_SRCS:.cu=.o)

# Build rules
all: $(TARGET)

$(TARGET): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(CPP_OBJS) $(CU_OBJS) $(TARGET)

.PHONY: all clean

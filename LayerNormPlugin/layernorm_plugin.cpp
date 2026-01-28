#include "layernorm_plugin.h"
#include "cuda_fp16.h"
#include <cstring>
#include <iostream>

// 使用外部kernel声明
extern "C" {
    void launch_layer_norm_float(
        float* output,
        const float* input,
        const float* gamma,
        const float* beta,
        float epsilon,
        int rows,
        int cols,
        cudaStream_t stream
    );
    
    void launch_layer_norm_half(
        __half* output,
        const __half* input,
        const __half* gamma,
        const __half* beta,
        float epsilon,
        int rows,
        int cols,
        cudaStream_t stream
    );
}

namespace {
    const char* LAYERNORM_PLUGIN_VERSION{"1"};
    const char* LAYERNORM_PLUGIN_NAME{"LayerNormPluginDynamic"};
}

// LayerNormPluginDynamic 实现
LayerNormPluginDynamic::LayerNormPluginDynamic(float epsilon)
    : mEpsilon(epsilon)
    , mPluginNamespace("")
    , mRows(0)
    , mCols(0)
    , mDataType(nvinfer1::DataType::kFLOAT) {}

LayerNormPluginDynamic::LayerNormPluginDynamic(const void* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    mEpsilon = *reinterpret_cast<const float*>(d);
    d += sizeof(float);
    mRows = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    mCols = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    mDataType = static_cast<nvinfer1::DataType>(*reinterpret_cast<const int*>(d));
    d += sizeof(int);
}

nvinfer1::IPluginV2DynamicExt* LayerNormPluginDynamic::clone() const noexcept {
    auto* plugin = new LayerNormPluginDynamic(mEpsilon);
    plugin->mRows = mRows;
    plugin->mCols = mCols;
    plugin->mDataType = mDataType;
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs LayerNormPluginDynamic::getOutputDimensions(
    int outputIndex, 
    const nvinfer1::DimsExprs* inputs, 
    int nbInputs, 
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    // 输出维度与输入维度相同
    return inputs[0];
}

bool LayerNormPluginDynamic::supportsFormatCombination(
    int pos, 
    const nvinfer1::PluginTensorDesc* inOut, 
    int nbInputs, 
    int nbOutputs) noexcept {
    // pos 0: input tensor
    // pos 1: gamma (scale)
    // pos 2: beta (bias)
    // pos 3: output tensor
    
    const auto& desc = inOut[pos];
    
    // Check data type - support FP32 and FP16
    if (desc.type != nvinfer1::DataType::kFLOAT && 
        desc.type != nvinfer1::DataType::kHALF) {
        return false;
    }
    
    // Check format - only LINEAR format supported
    if (desc.format != nvinfer1::PluginFormat::kLINEAR) {
        return false;
    }
    
    // All tensors (input, gamma, beta, output) must have the same data type
    if (pos > 0) {
        return desc.type == inOut[0].type;
    }
    
    return true;
}

void LayerNormPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, 
    int nbInputs, 
    const nvinfer1::DynamicPluginTensorDesc* out, 
    int nbOutputs) noexcept {
    // 检查输入数量
    if (nbInputs != 3) {
        std::cerr << "LayerNormPluginDynamic requires 3 inputs: input, gamma, beta" << std::endl;
        return;
    }
    
    // 获取特征维度
    const auto& inputDesc = in[0].desc;
    mCols = inputDesc.dims.d[inputDesc.dims.nbDims - 1];
    mDataType = in[0].desc.type;
}

size_t LayerNormPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, 
    int nbInputs, 
    const nvinfer1::PluginTensorDesc* outputs, 
    int nbOutputs) const noexcept {
    return 0;  // 不需要额外workspace
}

int LayerNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc, 
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, 
    void* const* outputs, 
    void* workspace, 
    cudaStream_t stream) noexcept {
    // 获取输入输出指针
    const void* input = inputs[0];
    const void* gamma = inputs[1];
    const void* beta = inputs[2];
    void* output = outputs[0];
    
    // 计算批次维度
    nvinfer1::Dims inputDims = inputDesc[0].dims;
    mRows = 1;
    for (int i = 0; i < inputDims.nbDims - 1; ++i) {
        mRows *= inputDims.d[i];
    }
    
    // 根据数据类型调用不同的kernel
    if (mDataType == nvinfer1::DataType::kFLOAT) {
        launch_layer_norm_float(
            static_cast<float*>(output),
            static_cast<const float*>(input),
            static_cast<const float*>(gamma),
            static_cast<const float*>(beta),
            mEpsilon,
            mRows,
            mCols,
            stream
        );
    } else if (mDataType == nvinfer1::DataType::kHALF) {
        launch_layer_norm_half(
            static_cast<__half*>(output),
            static_cast<const __half*>(input),
            static_cast<const __half*>(gamma),
            static_cast<const __half*>(beta),
            mEpsilon,
            mRows,
            mCols,
            stream
        );
    } else {
        std::cerr << "Unsupported data type in LayerNormPluginDynamic" << std::endl;
        return -1;
    }
    
    return 0;
}

nvinfer1::DataType LayerNormPluginDynamic::getOutputDataType(
    int index, 
    const nvinfer1::DataType* inputTypes, 
    int nbInputs) const noexcept {
    return inputTypes[0];
}

void LayerNormPluginDynamic::attachToContext(
    cudnnContext* cudnn, 
    cublasContext* cublas, 
    nvinfer1::IGpuAllocator* allocator) noexcept {}

void LayerNormPluginDynamic::detachFromContext() noexcept {}

const char* LayerNormPluginDynamic::getPluginType() const noexcept {
    return LAYERNORM_PLUGIN_NAME;
}

const char* LayerNormPluginDynamic::getPluginVersion() const noexcept {
    return LAYERNORM_PLUGIN_VERSION;
}

int LayerNormPluginDynamic::getNbOutputs() const noexcept {
    return 1;
}

int LayerNormPluginDynamic::initialize() noexcept {
    return 0;
}

void LayerNormPluginDynamic::terminate() noexcept {}

size_t LayerNormPluginDynamic::getSerializationSize() const noexcept {
    return sizeof(float) + sizeof(int) * 3;
}

void LayerNormPluginDynamic::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    
    // 序列化参数
    *reinterpret_cast<float*>(d) = mEpsilon;
    d += sizeof(float);
    
    *reinterpret_cast<int*>(d) = mRows;
    d += sizeof(int);
    
    *reinterpret_cast<int*>(d) = mCols;
    d += sizeof(int);
    
    *reinterpret_cast<int*>(d) = static_cast<int>(mDataType);
}

void LayerNormPluginDynamic::destroy() noexcept {
    delete this;
}

void LayerNormPluginDynamic::setPluginNamespace(const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* LayerNormPluginDynamic::getPluginNamespace() const noexcept {
    return mPluginNamespace.c_str();
}

// LayerNormPluginDynamicCreator 实现
nvinfer1::PluginFieldCollection LayerNormPluginDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField> LayerNormPluginDynamicCreator::mPluginAttributes;

LayerNormPluginDynamicCreator::LayerNormPluginDynamicCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("epsilon", nullptr, 
        nvinfer1::PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LayerNormPluginDynamicCreator::getPluginName() const noexcept {
    return LAYERNORM_PLUGIN_NAME;
}

const char* LayerNormPluginDynamicCreator::getPluginVersion() const noexcept {
    return LAYERNORM_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* LayerNormPluginDynamicCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV2* LayerNormPluginDynamicCreator::createPlugin(
    const char* name, 
    const nvinfer1::PluginFieldCollection* fc) noexcept {
    float epsilon = 1e-6f;
    
    // 解析插件字段
    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fc->fields[i].name;
        if (!strcmp(attrName, "epsilon")) {
            epsilon = *static_cast<const float*>(fc->fields[i].data);
        }
    }
    
    LayerNormPluginDynamic* plugin = new LayerNormPluginDynamic(epsilon);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

nvinfer1::IPluginV2* LayerNormPluginDynamicCreator::deserializePlugin(
    const char* name, 
    const void* serialData, 
    size_t serialLength) noexcept {
    LayerNormPluginDynamic* plugin = new LayerNormPluginDynamic(serialData, serialLength);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

void LayerNormPluginDynamicCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* LayerNormPluginDynamicCreator::getPluginNamespace() const noexcept {
    return mPluginNamespace.c_str();
}

// 注册插件
REGISTER_TENSORRT_PLUGIN(LayerNormPluginDynamicCreator);
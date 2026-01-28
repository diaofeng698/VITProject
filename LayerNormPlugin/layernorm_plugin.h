#pragma once

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferPlugin.h"
#include <string>
#include <vector>

class LayerNormPluginDynamic : public nvinfer1::IPluginV2DynamicExt {
public:
    LayerNormPluginDynamic(float epsilon = 1e-6f);
    LayerNormPluginDynamic(const void* data, size_t length);
    ~LayerNormPluginDynamic() override = default;
    
    // IPluginV2DynamicExt 方法
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, 
        const nvinfer1::DimsExprs* inputs, 
        int nbInputs, 
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, 
        const nvinfer1::PluginTensorDesc* inOut, 
        int nbInputs, 
        int nbOutputs) noexcept override;
    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, 
        int nbInputs, 
        const nvinfer1::DynamicPluginTensorDesc* out, 
        int nbOutputs) noexcept override;
    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs, 
        int nbInputs, 
        const nvinfer1::PluginTensorDesc* outputs, 
        int nbOutputs) const noexcept override;
    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc, 
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, 
        void* const* outputs, 
        void* workspace, 
        cudaStream_t stream) noexcept override;
    
    // IPluginV2Ext 方法
    nvinfer1::DataType getOutputDataType(
        int index, 
        const nvinfer1::DataType* inputTypes, 
        int nbInputs) const noexcept override;
    void attachToContext(
        cudnnContext* cudnn, 
        cublasContext* cublas, 
        nvinfer1::IGpuAllocator* allocator) noexcept override;
    void detachFromContext() noexcept override;
    
    // IPluginV2 方法
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    
private:
    float mEpsilon;
    std::string mPluginNamespace;
    int mRows;      // 批次维度
    int mCols;      // 特征维度
    nvinfer1::DataType mDataType;
};

class LayerNormPluginDynamicCreator : public nvinfer1::IPluginCreator {
public:
    LayerNormPluginDynamicCreator();
    ~LayerNormPluginDynamicCreator() override = default;
    
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(
        const char* name, 
        const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, 
        const void* serialData, 
        size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    
private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mPluginNamespace;
};
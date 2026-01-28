#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import torch
import tensorrt as trt
import numpy as np
import ctypes
import math
import time
import os

from typing import Optional, Tuple

import pycuda.driver as cuda
import pycuda.autoinit

class TrtNetworkHelper():
    """TensorRT Network Definition helper for Pytorch"""
    def __init__(self, network, plugin_registry, logger):
        self.network = network
        self.logger = logger
        self.input_num = 0
        self.plugin_registry = plugin_registry
        self.load_layer_norm_plugin()
        
    def set_layer_name(self, layer, name):
        """
        Tool function. Set the name of trt layer or plugin and print output shapes.
        """
        if not layer:
            raise RuntimeError("Could not name")

        layer.name = str(self.network.num_layers) + "_" + name
        for i in range(0, layer.num_outputs):
            shape = layer.get_output(i).shape
            self.logger.log(trt.Logger.INFO, "[Network] " + layer.name + ", output[" + str(i) + "] shape= " + str(shape))

        return None

    def check_trt_layer(self, trt_layer):
        """
        Tool function. check trt layer,
        """
        if not trt_layer:
            raise RuntimeError("add " + str(trt_layer) + " failed!")

        for i in range(0, trt_layer.num_outputs):
            shape = trt_layer.get_output(i).shape
            # print(trt.volume(shape))

            # if len(shape) is 1:
                # raise RuntimeError("add " + layer.name + " failed!")

    def layer_post_process(self, trt_layer, layer_name, precision):
        """
        Tool function. set precision, set_layer_name and check_trt_layer
        """
        if precision is not None:
            trt_layer.precision = precision

        self.set_layer_name(trt_layer, layer_name)
        self.check_trt_layer(trt_layer)

    def addInput(self, name, dtype, shape):
        if name is None:
            name = "input" + str(self.input_num)

        self.input_num = self.input_num + 1

        trt_input = self.network.add_input(name=name, dtype=dtype, shape=shape)
        if not trt_input:
            raise RuntimeError("addInput failed!")

        self.logger.log(trt.Logger.INFO, "[Network] add input:" + name + ", shape=" + str(shape))

        return trt_input

    def markOutput(self, x: trt.ITensor):
        self.network.mark_output(x)
        self.logger.log(trt.Logger.INFO, "[Network] mark output:" + x.name + ", shape=" + str(x.shape))

    def addConv2d(self, x, weight, bias, out_channels, kernel_size, stride=None, padding=None, dilation=None, groups=None,
                  layer_name=None, precision=None):
        """ConvCommon"""
        if layer_name is None:
            layer_name = "nn.Conv2d"
        else:
            layer_name = "nn.Conv2d." + layer_name

        weight = trt.Weights(weight)
        bias = trt.Weights(bias) if bias is not None else None

        trt_layer = self.network.add_convolution_nd(
            x, num_output_maps=out_channels,
            kernel_shape=kernel_size,
            kernel=weight, bias=bias)

        if stride is not None:
            trt_layer.stride = stride
        if padding is not None:
            trt_layer.padding = padding
        if dilation is not None:
            trt_layer.dilation = dilation
        if groups is not None:
            trt_layer.num_groups = groups

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addEmbedding(self, indices, weight, layer_name=None, precision=None):
        constant_layer = self.network.add_constant(weight.shape, trt.Weights(weight))
        gather_layer = self.network.add_gather(constant_layer.get_output(0),
                                               indices, axis=0)

        if layer_name is None:
            layer_name = "nn.Embedding"
        else:
            layer_name = "nn.Embedding." + layer_name

        self.layer_post_process(gather_layer, layer_name, precision)

        return gather_layer.get_output(0)

    def addGELU(self, x, layer_name=None, precision=None):
        POW = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
        MULTIPLY = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
        SQRT = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
        ONE = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
        HALF = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
        X_pow = self.network.add_elementwise(x, POW.get_output(0), trt.ElementWiseOperation.POW)
        X_pow_t = X_pow.get_output(0)
        X_mul = self.network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
        X_add = self.network.add_elementwise(x, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
        X_sqrt = self.network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
        X_sqrt_tensor = X_sqrt.get_output(0)
        X_tanh = self.network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
        X_tanh_tensor = X_tanh.get_output(0)
        X_one = self.network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
        CDF = self.network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
        gelu_layer = self.network.add_elementwise(CDF.get_output(0), x, trt.ElementWiseOperation.PROD)

        if layer_name is None:
            layer_name = "nn.GELU"
        else:
            layer_name = "nn.GELU." + layer_name

        self.layer_post_process(gelu_layer, layer_name, precision)

        return gelu_layer.get_output(0)

    def load_layer_norm_plugin(self, plugin_path: str = "./LayerNormPlugin/liblayernorm_plugin.so"):
        """Load LayerNorm plugin library"""
        if not os.path.exists(plugin_path):
            raise FileNotFoundError(f"Plugin library {plugin_path} does not exist")
        
        # Load plugin library
        ctypes.CDLL(plugin_path)
        
        # Find plugin creator
        plugin_creator = self.plugin_registry.get_plugin_creator("LayerNormPluginDynamic", "1")
        if plugin_creator is None:
            raise RuntimeError("LayerNormPluginDynamic not found, please ensure the plugin is compiled and registered correctly")
        
        self.layer_norm_creator = plugin_creator
        print("LayerNorm plugin loaded successfully")
        
    def addLayerNorm(self, x, gamma, beta, layer_name=None, eps=1e-5):
        """
        Add LayerNorm layer to network
        
        Args:
            network: TRT network
            x: Input tensor
            gamma: Scale parameter (shape: [hidden_size])
            beta: Bias parameter (shape: [hidden_size])
            layer_name: Layer name
            eps: Small constant to prevent division by zero
            
            
        Returns:
            Output tensor
        """
        # Get input shape
        input_shape = tuple(x.shape)
        hidden_size = input_shape[-1]
        
        # Validate parameter shapes
        if gamma.shape[0] != hidden_size:
            raise ValueError(f"gamma shape {gamma.shape} does not match input hidden dimension {hidden_size}")
        if beta.shape[0] != hidden_size:
            raise ValueError(f"beta shape {beta.shape} does not match input hidden dimension {hidden_size}")
        
        # Determine data type based on input tensor
        dtype = trt.float32 if x.dtype == trt.float32 else trt.float16
        np_dtype = np.float32 if dtype == trt.float32 else np.float16
        
        # Create constant layers for gamma and beta with matching dtype
        gamma_np = gamma.astype(np_dtype)
        beta_np = beta.astype(np_dtype)
        
        gamma_constant = self.network.add_constant(gamma_np.shape, gamma_np)
        gamma_constant.name = f"{layer_name}_gamma"
        gamma_tensor = gamma_constant.get_output(0)
        gamma_tensor.dtype = dtype
        
        beta_constant = self.network.add_constant(beta_np.shape, beta_np)
        beta_constant.name = f"{layer_name}_beta"
        beta_tensor = beta_constant.get_output(0)
        beta_tensor.dtype = dtype
        
        # Create plugin fields
        epsilon_field = trt.PluginField("epsilon", np.array([eps], dtype=np.float32), 
                                       trt.PluginFieldType.FLOAT32)
        plugin_fields = trt.PluginFieldCollection([epsilon_field])
        
        # Create plugin
        layer_norm_plugin = self.layer_norm_creator.create_plugin(layer_name, plugin_fields)
        if layer_norm_plugin is None:
            raise RuntimeError("LayerNorm plugin creation failed")
        
        # Create plugin layer
        plugin_layer = self.network.add_plugin_v2(
            [x, gamma_tensor, beta_tensor], 
            layer_norm_plugin
        )
        plugin_layer.name = layer_name
        
        output = plugin_layer.get_output(0)
        output.dtype = dtype
        
        return output
    
    def addLinear(self, x, weight, bias, layer_name=None, precision=None):
        input_len = len(x.shape)
        if input_len < 3:
            raise RuntimeError("addLinear x.shape.size must >= 3")
        
        if layer_name is None:
            layer_name = "nn.Linear"
        
        # calc pre_reshape_dims and after_reshape_dims
        pre_reshape_dims = trt.Dims()
        after_reshape_dims = trt.Dims()
        if input_len == 3:
            pre_reshape_dims = (0, 0, 0, 1, 1)
            after_reshape_dims = (0, 0, 0)
        elif input_len == 4:
            pre_reshape_dims = (0, 0, 0, 0, 1, 1)
            after_reshape_dims = (0, 0, 0, 0)
        elif input_len == 5:
            pre_reshape_dims = (0, 0, 0, 0, 0, 1, 1)
            after_reshape_dims = (0, 0, 0, 0, 0)
        else:
            raise RuntimeError("addLinear x.shape.size >5 not support!")
        
        # add pre_reshape layer
        trt_layer = self.network.add_shuffle(x)
        trt_layer.reshape_dims = pre_reshape_dims

        self.layer_post_process(trt_layer, layer_name+"_pre_reshape", precision)
        x = trt_layer.get_output(0)

        # add Linear layer
        out_features = weight.shape[0]
        weight = trt.Weights(weight)
        if bias is not None:
            bias = trt.Weights(bias)

        trt_layer = self.network.add_fully_connected(x, out_features, weight, bias)
        self.layer_post_process(trt_layer, layer_name, precision)
        x = trt_layer.get_output(0)

        # add after_reshape layer
        trt_layer = self.network.add_shuffle(x)
        trt_layer.reshape_dims = after_reshape_dims
        self.layer_post_process(trt_layer, layer_name+"_after_reshape", precision)
        x = trt_layer.get_output(0)

        return x
    
    def addReshape(self, x, reshape_dims, layer_name=None, precision=None):
        trt_layer = self.network.add_shuffle(x)
        trt_layer.reshape_dims = reshape_dims

        if layer_name is None:
            layer_name = "torch.reshape"
        else:
            layer_name = "torch.reshape." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x

    def addSlice(self, x, start_dim, shape_dim, stride_dim, layer_name=None, precision=None):
        trt_layer = self.network.add_slice(x, start_dim, shape_dim, stride_dim)

        if layer_name is None:
            layer_name = "tensor.slice"
        else:
            layer_name = "tensor.slice." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x

    def addReLU(self, layer, x, layer_name=None, precision=None):
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.RELU)

        if layer_name is None:
            layer_name = "nn.ReLU"

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addSoftmax(self, x: trt.ITensor, dim: int = -1, layer_name=None, precision=None) -> trt.ITensor:
        trt_layer = self.network.add_softmax(x)
        
        input_len = len(x.shape)
        if dim == -1:
            dim = input_len
        trt_layer.axes = int(math.pow(2, input_len-1))
        
        layer_name_prefix = "nn.Softmax[dim=" + str(dim) + "]"
        if layer_name is None:
            layer_name = layer_name_prefix
        else:
            layer_name = layer_name_prefix + "." + layer_name
       
        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addTanh(self, x, layer_name=None, precision=None):
        """Tanh"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.TANH)

        if layer_name is None:
            layer_name = "nn.Tanh"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    ################## unary op ###################
    def addLog(self, x: trt.ITensor, layer_name=None, precision=None):
        trt_layer = self.network.add_unary(x, trt.UnaryOperation.LOG)
        if layer_name is None:
            layer_name = "unary.log"
        else:
            layer_name = "unary.log." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    ################## elementwise op ###################
    def addAdd(self, a, b, layer_name=None, precision=None):
        trt_layer = self.network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
        if layer_name is None:
            layer_name = "elementwise.sum"
        else:
            layer_name = "elementwise.sum." + layer_name
        
        self.layer_post_process(trt_layer, layer_name, precision)
        
        x = trt_layer.get_output(0)
        return x

    # tensor and scalar op
    def addScale(
            self,
            x: trt.ITensor,
            scale: float,
            layer_name: str = None,
            precision: trt.DataType = None
    ) -> trt.ITensor:
        """scale"""
        input_len = len(x.shape)
        if input_len < 3:
            raise RuntimeError("input_len < 3 not support now! ")
        
        if layer_name is None:
            layer_name = "Scale"
        
        # The input dimension must be greater than or equal to 4
        if input_len is 3:
            trt_layer = self.network.add_shuffle(x)
            trt_layer.reshape_dims = (0, 0, 0, 1)
            self.layer_post_process(trt_layer, layer_name+".3dto4d", precision)
            x = trt_layer.get_output(0)
        
        np_scale = trt.Weights(np.array([scale], dtype=np.float32))
        trt_layer = self.network.add_scale(x, mode=trt.ScaleMode.UNIFORM,shift=None, scale=np_scale, power=None)
        self.layer_post_process(trt_layer, layer_name, precision)
        x = trt_layer.get_output(0)
        
        if input_len is 3:
            trt_layer = self.network.add_shuffle(x)
            trt_layer.reshape_dims = (0, 0, 0)
            self.layer_post_process(trt_layer, layer_name+".4dto3d", precision)
            x = trt_layer.get_output(0)
        
        return x

    def addCat(self, inputs = [], dim = 0, layer_name=None, precision=None):
        assert len(inputs) > 1

        trt_layer = self.network.add_concatenation(inputs)

        if dim == -1:
            dim = len(inputs[0].shape) - 1

        trt_layer.axis = dim

        if layer_name is None:
            layer_name = "torch.cat"
        else:
            layer_name = "torch.cat." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addMatMul(self, a: trt.ITensor, b: trt.ITensor, layer_name: Optional[str] = None) -> trt.ITensor:
        
        trt_layer = self.network.add_matrix_multiply(a, trt.MatrixOperation.NONE,b, trt.MatrixOperation.NONE)
        if layer_name is None:
            layer_name = "torch.matmul"
        else:
            layer_name = "torch.matmul." + layer_name
        
        self.layer_post_process(trt_layer, layer_name, None)
        x = trt_layer.get_output(0)
        return x

    def addConstant(self, w, layer_name: Optional[str] = None) -> trt.ITensor:
        trt_layer = self.network.add_constant(w.shape, w)

        if layer_name is None:
            layer_name = "trt.Constant"
        else:
            layer_name = "trt.Constant." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)
        x = trt_layer.get_output(0)
        return x

    def addShuffle(
        self,
        x: trt.ITensor,
        first_transpose: trt.Permutation,
        reshape_dims: trt.Dims,
        second_transpose: trt.Permutation,
        layer_name: Optional[str] = None
    ) -> trt.ITensor:
        """"""
        trt_layer = self.network.add_shuffle(x)
        if first_transpose is not None:
            trt_layer.first_transpose = first_transpose

        if reshape_dims is not None:
            trt_layer.reshape_dims = reshape_dims

        if second_transpose is not None:
            trt_layer.second_transpose = second_transpose

        if layer_name is None:
            layer_name = "trt.Shuffle"
        else:
            layer_name = "trt.Shuffle." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x


class InferHelper():
    """"""
    def __init__(self, plan_name, trt_logger):
        """"""
        self.logger = trt_logger
        self.runtime = trt.Runtime(trt_logger)
        with open(plan_name, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.context.active_optimization_profile = 0

    def infer(self, inputs: list, is_log = False):
        nInput = len(inputs)

        bufferD = []
        # alloc memory
        for i in range(nInput):
            bufferD.append(cuda.mem_alloc(inputs[i].nbytes))
            cuda.memcpy_htod(bufferD[i], inputs[i].ravel())
            self.context.set_binding_shape(i, tuple(inputs[i].shape))
            # print(inputs[i].nbytes)

        # for i in range(0, self.engine.num_bindings):
            # print("get_binding_shape:" + str(self.context.get_binding_shape(i)))

        outputs = []
        for i in range(len(inputs), self.engine.num_bindings):
            outputs.append(np.zeros(self.context.get_binding_shape(i)).astype(np.float32))

        nOutput = len(outputs)
        for i in range(nOutput):
            bufferD.append(cuda.mem_alloc(outputs[i].nbytes))
            # print(outputs[i].nbytes)

        for i in range(len(inputs), self.engine.num_bindings):
            trt_output_shape = self.context.get_binding_shape(i)
            output_idx = i - len(inputs)
            if not (list(trt_output_shape) == list(outputs[output_idx].shape)):
                self.logger.log(trt.Logger.ERROR, "[Infer] output shape is error!")
                self.logger.log(trt.Logger.ERROR, "trt_output.shape = " + str(trt_output_shape))
                self.logger.log(trt.Logger.ERROR, "base_output.shape = " + str(outputs[output_idx].shape))
                assert(0)

        # warm up
        self.context.execute_v2(bufferD)

        T1 = time.perf_counter()

        self.context.execute_v2(bufferD)

        T2 =time.perf_counter()
        if is_log:
            print("time=" + str((T2-T1) * 1000) + "ms")

        for i in range(nInput, nInput + nOutput):
            cuda.memcpy_dtoh(outputs[i - nInput].ravel(), bufferD[i])

        if is_log:
            for i in range(0, len(outputs)):
                print("outputs.shape:" + str(outputs[i].shape))
                print("outputs.sum:" + str(outputs[i].sum()))
                print(outputs[i])

            # print("trt_output.shape:" + str(trt_output.shape))
            # print("trt_output.sum:" + str(trt_output.sum()))
            # print(trt_output.view(-1)[0:10])
            # print("torch.allclose result:" + str(torch.allclose(base_output, trt_output, 1e-05, 1e-03)))
            # print("====================")
        return outputs
        # return torch.allclose(base_output, trt_output, 1e-05, 1e-03)

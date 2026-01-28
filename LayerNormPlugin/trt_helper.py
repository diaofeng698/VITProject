import tensorrt as trt
import numpy as np
import ctypes
import os
from typing import Optional, List, Tuple

class TRTBuilder:
    """TensorRT builder helper class"""
    
    def __init__(self, logger_level=trt.Logger.ERROR):
        self.logger = trt.Logger(logger_level)
        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        
        # Set optimization configuration
        self.config.set_flag(trt.BuilderFlag.FP16)
        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        self.config.max_workspace_size = 1 << 30  # 1GB
        
        # Plugin registration
        self.plugin_registry = trt.get_plugin_registry()
        self.load_layer_norm_plugin()
    
    def load_layer_norm_plugin(self, plugin_path: str = "./liblayernorm_plugin.so"):
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
    
    def add_layer_norm_layer(
        self,
        network: trt.INetworkDefinition,
        input_tensor: trt.ITensor,
        gamma: np.ndarray,
        beta: np.ndarray,
        epsilon: float = 1e-6,
        name: str = "layer_norm"
    ) -> trt.ITensor:
        """
        Add LayerNorm layer to network
        
        Args:
            network: TRT network
            input_tensor: Input tensor
            gamma: Scale parameter (shape: [hidden_size])
            beta: Bias parameter (shape: [hidden_size])
            epsilon: Small constant to prevent division by zero
            name: Layer name
            
        Returns:
            Output tensor
        """
        # Get input shape
        input_shape = tuple(input_tensor.shape)
        hidden_size = input_shape[-1]
        
        # Validate parameter shapes
        if gamma.shape[0] != hidden_size:
            raise ValueError(f"gamma shape {gamma.shape} does not match input hidden dimension {hidden_size}")
        if beta.shape[0] != hidden_size:
            raise ValueError(f"beta shape {beta.shape} does not match input hidden dimension {hidden_size}")
        
        # Determine data type based on input tensor
        dtype = trt.float32 if input_tensor.dtype == trt.float32 else trt.float16
        np_dtype = np.float32 if dtype == trt.float32 else np.float16
        
        # Create constant layers for gamma and beta with matching dtype
        gamma_np = gamma.astype(np_dtype)
        beta_np = beta.astype(np_dtype)
        
        gamma_constant = network.add_constant(gamma_np.shape, gamma_np)
        gamma_constant.name = f"{name}_gamma"
        gamma_tensor = gamma_constant.get_output(0)
        gamma_tensor.dtype = dtype
        
        beta_constant = network.add_constant(beta_np.shape, beta_np)
        beta_constant.name = f"{name}_beta"
        beta_tensor = beta_constant.get_output(0)
        beta_tensor.dtype = dtype
        
        # Create plugin fields
        epsilon_field = trt.PluginField("epsilon", np.array([epsilon], dtype=np.float32), 
                                       trt.PluginFieldType.FLOAT32)
        plugin_fields = trt.PluginFieldCollection([epsilon_field])
        
        # Create plugin
        layer_norm_plugin = self.layer_norm_creator.create_plugin(name, plugin_fields)
        if layer_norm_plugin is None:
            raise RuntimeError("LayerNorm plugin creation failed")
        
        # Create plugin layer
        plugin_layer = network.add_plugin_v2(
            [input_tensor, gamma_tensor, beta_tensor], 
            layer_norm_plugin
        )
        plugin_layer.name = name
        
        output = plugin_layer.get_output(0)
        output.dtype = dtype
        
        return output
    
    def _create_constant_tensor(
        self,
        network: trt.INetworkDefinition,
        data: np.ndarray,
        name: str
    ) -> trt.ITensor:
        """Create constant tensor"""
        # Create weights
        weight = np.ascontiguousarray(data.astype(np.float32))
        weight_tensor = network.add_constant(weight.shape, weight)
        weight_tensor.name = name
        return weight_tensor.get_output(0)
    
    def build_engine(
        self,
        network: trt.INetworkDefinition,
        max_batch_size: int = 1
    ) -> trt.ICudaEngine:
        """Build TensorRT engine"""
        # Set network input
        profile = self.builder.create_optimization_profile()
        
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_shape = input_tensor.shape
            
            # Create optimization profile
            if len(input_shape) >= 1:
                min_shape = (1,) + input_shape[1:]
                opt_shape = (max_batch_size // 2,) + input_shape[1:]
                max_shape = (max_batch_size,) + input_shape[1:]
            else:
                min_shape = (1,)
                opt_shape = (max_batch_size // 2,)
                max_shape = (max_batch_size,)
            
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        
        self.config.add_optimization_profile(profile)
        
        # Build engine
        engine = self.builder.build_engine(network, self.config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed")
        
        return engine
    
    def save_engine(self, engine: trt.ICudaEngine, path: str):
        """Save engine to file"""
        with open(path, "wb") as f:
            f.write(engine.serialize())
        print(f"Engine saved to: {path}")
    
    def load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """Load engine from file"""
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

# Usage example
def example_build_network():
    """Build example network with LayerNorm"""
    
    # Initialize builder
    builder = TRTBuilder()
    
    # Create network
    network = builder.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Define input
    input_tensor = network.add_input("input", trt.float32, (-1, 512))  # Dynamic shape
    
    # Define LayerNorm parameters
    hidden_size = 512
    gamma = np.ones((hidden_size,), dtype=np.float32)  # Scale parameter
    beta = np.zeros((hidden_size,), dtype=np.float32)  # Bias parameter
    
    # Add LayerNorm layer
    output = builder.add_layer_norm_layer(
        network=network,
        input_tensor=input_tensor,
        gamma=gamma,
        beta=beta,
        epsilon=1e-6,
        name="layer_norm"
    )
    
    # Set output
    output.name = "output"
    network.mark_output(output)
    
    # Set output data type
    output.dtype = trt.float32
    
    return builder, network

# Test function
def test_layer_norm_plugin():
    """Test LayerNorm plugin"""
    import torch
    import torch.nn.functional as F
    
    print("=== TensorRT LayerNorm Plugin Test ===")
    
    try:
        # Build network
        builder, network = example_build_network()
        
        # Build engine
        print("Building TensorRT engine...")
        engine = builder.build_engine(network, max_batch_size=32)
        print("Engine built successfully")
        
        # Save engine
        builder.save_engine(engine, "layer_norm_engine.trt")
        
        # Compare with PyTorch
        print("\n=== Comparison Test with PyTorch ===")
        
        # Create test data
        batch_size = 4
        hidden_size = 512
        input_np = np.random.randn(batch_size, hidden_size).astype(np.float32)
        
        # PyTorch calculation
        input_torch = torch.from_numpy(input_np)
        gamma = torch.ones(hidden_size)
        beta = torch.zeros(hidden_size)
        output_torch = F.layer_norm(input_torch, (hidden_size,), gamma, beta, eps=1e-6)
        print("PyTorch output calculated")
        print(f"PyTorch output sample: {output_torch[0, :5]}")
        
        # TensorRT calculation
        import pycuda.driver as cuda
        import pycuda.autoinit
        context = engine.create_execution_context()
        input_device = cuda.mem_alloc(input_np.nbytes)
        output_np = np.empty_like(input_np)
        output_device = cuda.mem_alloc(output_np.nbytes)
        cuda.memcpy_htod(input_device, input_np)
        context.set_binding_shape(0, (batch_size, hidden_size))
        bindings = [int(input_device), int(output_device)]
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(output_np, output_device)
        print("TensorRT output calculated")
        print(f"TensorRT output sample: {output_np[0, :5]}")
        
        # calculate Euclidean distance
        diff = np.linalg.norm(output_np - output_torch.numpy())
        print(f"Output difference (L2 norm): {diff}")

        return True

        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # First compile the plugin
    print("1. Compile plugin:")
    os.system("bash build.sh")
    print()
    
    # Test plugin
    print("2. Run test:")
    if os.path.exists("./liblayernorm_plugin.so"):
        test_layer_norm_plugin()
    else:
        print("Please compile the plugin library first: liblayernorm_plugin.so")
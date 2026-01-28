#!/bin/bash

# Define image name and tag
IMAGE_NAME="trt7.2.3_env"
TAG="v1.0"
DOCKERFILE="Dockerfile.trt723"

# check if local TensorRT file exists if you use the COPY method
# if [ ! -f "TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz" ]; then
#     echo "Warning: TensorRT tar file not found. Make sure to download it or adjust Dockerfile."
# fi

echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${TAG} -f ${DOCKERFILE} .

echo "Running Docker container..."
# --gpus all: Enable all GPUs
# -v $(pwd):/workspace: Mount current directory to /workspace inside container
# --shm-size=1g: Increase shared memory size (often needed for PyTorch/TRT)
docker run --gpus all -it \
    --shm-size=8g \
    -v $(pwd):/workspace \
    ${IMAGE_NAME}:${TAG}

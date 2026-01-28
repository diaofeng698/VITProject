import argparse
import ctypes
import numpy as np
import os
import pycuda.autoinit
import torch

from torchvision import transforms
from PIL import Image
import tensorrt as trt
from trt_helper import InferHelper

# TensorRT Initialization
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Load plugins
handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

handle = ctypes.CDLL("liblayernorm_plugin.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `liblayernorm_plugin.so` on your LD_LIBRARY_PATH?")

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def image_preprocess(path):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = transform(img)
    return img

def read_all_test_imgs(test_data_path):
    imgs = []
    labels = []
    label_idx = 0
    
    for c in classes:
        path = os.path.join(test_data_path, c)
        class_count = 0
        for i in os.listdir(path):
            img = image_preprocess(os.path.join(path, i))
            imgs.append(img)
            labels.append(label_idx)
            class_count += 1
        label_idx += 1
        print(f"Read {c} done, img num = {class_count}")

    return imgs, labels

def valid_trt(plan_path, imgs, labels):
    print("***** Running TensorRT Validation *****")
    
    infer_helper = InferHelper(plan_path, TRT_LOGGER)
    acc_count = 0
    
    for i in range(len(imgs)):
        img = torch.unsqueeze(imgs[i], 0).numpy()
        logits = infer_helper.infer([img])
        max_idx = np.argmax(logits[0])
        
        if max_idx == labels[i]:
            acc_count += 1

        if (i + 1) % 100 == 0:
            acc_rate = acc_count / (i + 1) * 100
            print(f"Processed {i+1} images, acc_count={acc_count}, acc_rate={acc_rate:.2f}%")

    acc_rate = acc_count / len(imgs) * 100
    print(f"Final accuracy = {acc_rate:.2f}%")

def main():
    parser = argparse.ArgumentParser(
        description="TensorRT ViT Validation", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-p", "--plan", required=True, help="The TensorRT engine file path")
    parser.add_argument("-d", "--test-data", required=True, help="Test data directory")

    args = parser.parse_args()

    imgs, labels = read_all_test_imgs(args.test_data)
    valid_trt(args.plan, imgs, labels)

if __name__ == "__main__":
    main()

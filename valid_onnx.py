import argparse
import numpy as np
import os
import torch
import onnxruntime as ort

from torchvision import transforms
from PIL import Image

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
    class_count = 0
    for c in classes:
        path = test_data_path + "/" + c
        for i in os.listdir(path):
            img = image_preprocess(path + "/" + i)
            imgs.append(img)
            labels.append(label_idx)
            class_count = class_count + 1
        label_idx = label_idx + 1
        print(f"read {c} done ..., img num = {class_count}")
        class_count = 0

    return imgs, labels


def valid_onnx(onnx_path, imgs, labels):
    print("***** Running ONNX Validation *****")

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    acc_count = 0
    for i in range(len(imgs)):
        img = torch.unsqueeze(imgs[i], 0).numpy()
        logits = ort_session.run(None, {'input': img})
        max_idx = np.argmax(logits[0])
        if max_idx == labels[i]:
            acc_count = acc_count + 1

        if (i+1) % 100 == 0:
            acc_rate = acc_count / (i+1) * 100
            print(f"acc_count={str(acc_count)}, acc_rate={str(acc_rate)}%")

    acc_rate = acc_count / len(imgs) * 100
    print(f"final acc = {str(acc_rate)}%")

def main():
    parser = argparse.ArgumentParser(description="ONNX Model Validation", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-x", "--onnx", required=True, help="The ONNX model file path.")
    parser.add_argument("-d", "--test-data", required=True, help="Valid test data directory")

    args, _ = parser.parse_known_args()

    imgs, labels = read_all_test_imgs(args.test_data)
    valid_onnx(args.onnx, imgs, labels)

if __name__ == "__main__":
    main()

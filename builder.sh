export LD_LIBRARY_PATH=./LayerNormPlugin/:$LD_LIBRARY_PATH
# python builder.py -x ViT-B_16.onnx -o model_plugin.plan --img_path ./CIFAR-10-images/test/cat/0000.jpg > logs/build.log 2>&1
# python builder.py -x ViT-B_16.onnx -o model_fp16.plan --img_path ./CIFAR-10-images/test/cat/0000.jpg -f > logs/build.log 2>&1
python builder.py -x ViT-B_16.onnx -o model_int8.plan --img_path ./CIFAR-10-images/test/cat/0000.jpg -i -p ./CIFAR-10-images/test -f > logs/build.log 2>&1
# DeepLearningFrameworks
The inference speed of different depth learning frameworks on CPU is compared. 不同的深度学习框架在cpu上的推理速度比较。

## 实验环境
系统： Windows 10
处理器： Intel(R) Core(TM) i7-7700 CPU @ 3.6GHz 3.60GHz
Python:  3.6.7 (default, Jul  2 2019, 02:21:41) [MSC v.1900 64 bit (AMD64)]
Numpy:  1.16.4
pytorch version: 1.1.0
caffe2 version: 1.1.0
tensorflow version: 1.9.0
onnxruntime version: 0.4.0


## 安装步骤
1. 安装anaconda,确定好python的版本，安装numpy.
2. 安装pytorch:`conda install pytorch`
3. 安装tensorflow: `conda install tensorflow`
4. 安装onnxruntime: `conda install onnxruntime`


## 实验结果
| DL Library                                            | resnet50 | resnet18 |
| ----------------------------------------------------- | :----------------: | :-----------------: |
| TensorFlow                 |        148         |         54          |
| PyTorch                |        162         |         69          |
| Caffe2                        |        163         |         53          |
| Onnxruntime             |        152         |         57          |
|Onnxruntime + mkldnn         |        194         |         76          |
| Onnxruntime + ngraph             |        241         |         76          |




# DeepLearningFrameworks
The inference speed of different depth learning frameworks on CPU is compared.   
对比resnet18和resnet50在不同的深度学习框架上推理速度。

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
1. 安装anaconda。
2. 创建一个新的环境`conda create -n dlf python=3.6.7`, 并激活环境`conda activate dlf`。
3. 安装numpy`conda install numpy=1.16.4`
4. 安装pytorch:`conda install pytorch-cpu torchvision-cpu -c pytorch`
5. 安装caffe2: 安装好pytorch之后会自带caffe2，但是要成功执行的话还需要安装一些依赖包，根据出错提示google一下就行。
6. 安装tensorflow: `conda install tensorflow=1.9.0`
7. 安装onnxruntime: `pip install onnxruntime==0.4.0`(如果要框架支持MKLDNN 或者 Ngraph加速器的话需要从源码安装：https://github.com/microsoft/onnxruntime/blob/master/BUILD.md)
8. 还需执行一下： `conda install pandas tqdm`

## 执行步骤
 1. 先将预训练模型下载到`./models/`中对应的文件下，下载地址在对应文件中有给出。（不用下载pytorch的预训练模型，具体的可以看pytorch_infer.py里面的代码。） 
 2. 执行对应的`*_infer.py`的文件即可在`./result/`文件夹中生成对应的实验结果。

 
## 实验结果
单位：fps  
batch size: 1  

| DL Library             | resnet50           | resnet18           |
| ---------------------- | :----------------: | :----------------: |
| TensorFlow             |        3.9         |         -          |
| PyTorch                |        5.7         |         11.3       |
| Caffe2                 |        14.6        |         -          |
| Onnxruntime            |        25.8        |         59.0       |
| Onnxruntime + mkldnn   |        25.4        |         71.6       |
| Onnxruntime + ngraph   |        36.1        |         89.8       |

结论：  
可以看出 Onnxruntime这个框架即使没有使用mkldnn加速的时候，也会比其他框架快很多，而且其他几个框架的模型都可以转成ONNX的格式。     
备注：  
1.TensorFlow 和 Caffe2 的官方resnet18的预训练模型没有找到。  
2.PyTorch 和 TensorFlow 目前还不支持在在Windows中使用MKLDNN加速,具体原因：[Pytorch](https://github.com/pytorch/pytorch/issues/22962) [TensorFlow](https://www.tensorflow.org/guide/performance/overview)

## 其他资料链接
1. TensorFlow + mkldnn: https://www.anaconda.com/tensorflow-in-anaconda/    
2. ONNX github： https://github.com/onnx/onnx  
3. onnxruntime github：https://github.com/microsoft/onnxruntime  
4. PyTorch github: https://github.com/pytorch/pytorch  

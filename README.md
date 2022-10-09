# Face_Detection_SSD<br>
real_time_face_detection_classify_smile_and_face_mosaic<br>
[数据集下载]链接：https://pan.baidu.com/s/1ZveCAMqAgS6r2CqShrXppA 
提取码：m2l5
<br>

环境：win10 + keras2.0 + tensorflow(1.9)


人脸微笑分类数据库树状结构如下：<br>
-----数据集----- <br>
>SMILEs<br>     微笑数据集<br>
>>\negatives\negatives\img.jpg......<br>     >>负样本数据集图片存储<br>
>>\positives\positives\img.jpg......<br>     >>正样本数据集图片存储<br>

文件路径结构如下：<br>
>face_detection_SSD<br>         基于SSD人脸检测源码<br>
>>|---train_ResNet.py<br>       残差网络，利用该网络训练残差模型来对人脸图片进行识别。<br>
>>|---SSD_detection_smile_masic.py<br>    基于SSD人脸微笑检测具备马赛克能力，适用于数据为视频格式。<br>
>>|---SSD_detection_smile_masic_image.py<br> 基于SSD人脸微笑检测具备马赛克能力，适用于数据为图片格式。<br>
>>|---Masic.py<br>   马赛克源码<br>
>>|---nn<br>      网络模型<br>
>>>|---__init__.py<br>
>>>|---resnet.py<br>   残差网络模型<br>
>>>|---lenet.py<br>    lenet网络模型<br>

微笑分类训练命令：<br>
```java
python train_ResNet.py -d 数据集路径 -m 模型保存路径   
```
例如：<br>
```java
python train_ResNet.py -d E:\ImageDataSet\SMILEs\ -m E:\ImageDataSet\SMILEs\resnet.hdf5 
```
摄像头实时人脸检测命令：<br>
```java
python SSD_detection_smile_masic.py -p SSD的deploy.prototxt文件路径 -m SSD模型路径 -cm 微笑分类器模型路径
```
例如<br>
```java
python SSD_detection_smile_masic.py -p E:\ImageDataSet\deep-learning-face-detection\deep-learning-face-detection\deploy.prototxt.txt -m E:\ImageDataSet\deep-learning-face-detection\deep-learning-face-detection\res10_300x300_ssd_iter_140000.caffemodel -cm E:\ImageDataSet\SMILEs\resnet.hdf5
```

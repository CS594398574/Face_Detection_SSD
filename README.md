# Face_Detection_SSD<br>
real_time_face_detection_classify_smile_and_face_mosaic<br>
[数据集下载]（https://github.com/hromi/SMILEsmileD）<br>

环境：win10 + keras2.0 + tensorflow(1.9)


人脸微笑分类数据库树状结构如下：<br>
-----数据集----- <br>
>SMILEs<br>
>>\negatives\negatives\img.jpg......<br>
>>\positives\positives\img.jpg......<br>

文件路径结构如下：<br>
>face_detection_SSD<br>
>>|---train_ResNet.py<br>
>>|---SSD_detection_smile_masic.py<br>
>>|---SSD_detection_smile_masic_image.py<br>
>>|---Masic.py<br>
>>|---nn<br>
>>>|---__init__.py<br>
>>>|---resnet.py<br>
>>>|---lenet.py<br>

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

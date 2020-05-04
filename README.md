# Face_Detection_SSD
real_time_face_detection_classify_smile_and_face_mosaic
人脸微笑分类数据库树状结构如下：
-----datasets----- <br>
|...\SMILEs
|...\SMILEs\negatives\negatives\img.jpg......
|...\SMILEs\positives\positives\img.jpg......

文件路径结构如下：
---face_detection_SSD
|   |---train_ResNet.py
|   |---SSD_detection_smile_masic.py
|   |---SSD_detection_smile_masic_image.py
|   |---Masic.py
|   |---nn
|   |    |---__init__.py
|   |    |---resnet.py
|   |    |---lenet.py



微笑分类训练命令：
python train_ResNet.py -d 数据集路径 -m 模型保存路径
例如：python train_ResNet.py -d E:\\ImageDataSet\\SMILEs\\ -m E:\\ImageDataSet\\SMILEs\\resnet.hdf5

摄像头实时人脸检测命令：
python SSD_detection_smile_masic.py -p SSD的deploy.prototxt文件路径 -m SSD模型路径 -cm 微笑分类器模型路径
例如：python SSD_detection_smile_masic.py -p E:\ImageDataSet\deep-learning-face-detection\deep-learning-face-detection\deploy.prototxt.txt -m E:\ImageDataSet\deep-learning-face-detection\deep-learning-face-detection\res10_300x300_ssd_iter_140000.caffemodel -cm E:\CVDL\Smiles\lenet.hdf5

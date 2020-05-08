from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from Masic import anonymize_face_pixelate,anoymize_face_simple
from keras.models import load_model
from keras.preprocessing.image import img_to_array
ap = argparse.ArgumentParser()
ap.add_argument("-p","--prototxt",required=True,
                help="Path to Caffe 'deploy' prototxt file")
ap.add_argument("-m","--model",required=True,
                help="Path to Caffe pre-trained model")
ap.add_argument("-c","--confidence",type=float,default=0.5,
                help="minimum probaility to filter weak detections")
ap.add_argument("-cm","--cmodel",required=True,
                help="Path to keras Classifier model for emotion")
ap.add_argument("-i","--image",required=True,
                help="Path to Image")
args = vars(ap.parse_args())


print("[INFO] loading model....")
net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])
cmodel = load_model(args["cmodel"])
frame = cv2.imread(args["image"])
frame = imutils.resize(frame,width=600)
frameClone = frame.copy()
#得到帧的维度，将帧转换成blob格式
(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
for i in range(0,detections.shape[2]):
    #抽取预测的置信度
    confidence = detections[0,0,i,2]
    #过滤低于最小阈值置信度的检测
    if confidence < args["confidence"]:
        continue
    #计算物体的边界框(x,y)坐标
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX,startY,endX,endY) = box.astype("int")
    roiclone = frameClone[startY:endY, startX:endX]
    roiclone = anonymize_face_pixelate(roiclone)
    frameClone[startY:endY, startX:endX] = roiclone
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[startY:endY, startX:endX]
    roi = cv2.resize(roi, (28, 28))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    (notSmiling, smiling) = cmodel.predict(roi)[0]
    label = "Smiling" if smiling > notSmiling else "Not Smiling"

    text ="{:.2f}%".format(confidence*100)
    y = startY -10 if startY -10 >10 else startY +10
    cv2.rectangle(frameClone,(startX,startY),(endX,endY),
                  (0,0,255),2)
    cv2.putText(frameClone, label, (endX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(frameClone,text,(startX,y),cv2.FONT_HERSHEY_COMPLEX,
                0.45,(0,0,255),1)
cv2.imshow("picture",frameClone)
cv2.waitKey(0)
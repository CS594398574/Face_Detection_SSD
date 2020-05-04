import cv2
import numpy as np

def anoymize_face_simple(image,factor=3.0): #进行高斯blur
    (h,w) = image.shape[:2]
    kW = int(w/factor)
    kH = int(h/factor)
    if kW %2==0:
        kW-=1
    if kH %2 == 0:
        kH-=1
    return cv2.GaussianBlur(image,(kW,kH),0)

def anonymize_face_pixelate(image,blocks=9):
    (h,w)= image.shape[:2]
    xSteps = np.linspace(0,w,blocks +1,dtype="int")
    ySteps = np.linspace(0,h,blocks +1,dtype="int")
    #将image分成  M*N  块
    for i in range(1,len(ySteps)):
        for j in range(1,len(xSteps)):
            startX =  xSteps[j-1]
            startY =  ySteps[i-1]
            endX = xSteps[j]
            endY = ySteps[i]

            roi = image[startY:endY,startX:endX]
            (B,G,R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image,(startX,startY),(endX,endY),(B,G,R),-1)
    return image

# example:
# house = cv2.imread("E:\\ImageDataSet\\timg.jpg")
#
# roi = house[:120,:60]
# gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
# gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
# gray = anonymize_face_pixelate(gray)
# house[:120,:60] = gray
# cv2.imshow("windows",house)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

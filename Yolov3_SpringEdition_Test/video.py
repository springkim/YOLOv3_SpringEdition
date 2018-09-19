import os
import re
import ctypes
import cv2
import platform
import random
import numpy as np
import urllib.request
if not os.path.isfile("paris.avi"):
	print("Downloading paris.avi")
	urllib.request.urlretrieve("https://github.com/springkim/YOLOv3_SpringEdition/releases/download/image/paris.avi","paris.avi")

dlname="libYOLOv3SE"
if platform.system()=='Windows':
    dlname+=".dll"
else:
    dlname+=".so"
yolov3_dl=ctypes.cdll.LoadLibrary(os.path.abspath(dlname))

yolov3_dl.YoloLoad.restype = ctypes.POINTER(ctypes.c_int)
net=yolov3_dl.YoloLoad("yolov3_darknet53_coco.cfg".encode('ascii'),"yolov3_darknet53.weights".encode('ascii'))

vc = cv2.VideoCapture('paris.avi')

with open('coco.names') as f:
    classes = f.read().splitlines()

def gen_colors(n):
  ret = []
  for i in range(n):
    r =int(random.random() * 256)% 256
    g = int(random.random() * 256) % 256
    b = int(random.random() * 256) % 256
    ret.append((r,g,b)) 
  return ret

colors=gen_colors(len(classes))

result=(ctypes.c_float*1024)()
while(vc.isOpened()):
    ret, frame=vc.read()
    if ret==True:
        fltimg=frame.astype(np.float32)
        h=frame.shape[0]
        w=frame.shape[1]
        c=frame.shape[2]

        yolov3_dl.YoloDetectFromImage.argtypes=[ctypes.POINTER(ctypes.c_float)]
        r=yolov3_dl.YoloDetectFromPyImage(fltimg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),w,h,c,net,ctypes.c_float(0.1),result,1024)
       

        for j in range(0,r-1):
            c=int(result[j*6+0])
            x1=int(result[j*6+2])
            y1=int(result[j*6+3])
            x2=int(result[j*6+4]+x1)
            y2=int(result[j*6+5]+y1)
            frame=cv2.rectangle(frame,(x1,y1),(x2,y2),colors[c],2)
            frame=cv2.putText(frame, classes[c], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[c], 2)
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cv2.destroyAllWindows()

yolov3_dl.YoloRelease(net)
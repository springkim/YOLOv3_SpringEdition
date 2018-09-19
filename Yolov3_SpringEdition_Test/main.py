import os
import re
import ctypes
import cv2
import platform
dlname="libYOLOv3SE"
if platform.system()=='Windows':
    dlname+=".dll"
else:
    dlname+=".so"
yolov3_dl=ctypes.cdll.LoadLibrary(os.path.abspath(dlname))

yolov3_dl.YoloLoad.restype = ctypes.POINTER(ctypes.c_int)
net=yolov3_dl.YoloLoad("yolov3_darknet53_coco.cfg".encode('ascii'),"yolov3_darknet53.weights".encode('ascii'))

files = [f for f in os.listdir('voc2007valid') if re.match(r'[0-9]+.*\.jpg', f)]
files.sort()

with open('coco.names') as f:
    classes = f.read().splitlines()

result=(ctypes.c_float*1024)()
for i in range(0,len(files)):
    img_file="voc2007valid/"+files[i]
    img=cv2.imread(img_file,cv2.IMREAD_COLOR)
    r=yolov3_dl.YoloDetectFromFile(img_file.encode('ascii'),net,ctypes.c_float(0.1),result,1024)
    for j in range(0,r-1):
        x1=int(result[j*6+2])
        y1=int(result[j*6+3])
        x2=int(result[j*6+4]+x1)
        y2=int(result[j*6+5]+y1)
        img=cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
        img=cv2.putText(img, classes[int(result[j*6+0])], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('result', img)
    if cv2.waitKey(0)==27:
	    break
cv2.destroyAllWindows()



yolov3_dl.YoloRelease(net)

from ctypes import *
libc = CDLL("yolo_v2_320.dll")
sz=libc.YoloSizePy()
network = create_string_buffer(sz)
libc.YoloLoadPy("../../network/yolo.cfg","../../network/yolo.weights",network)

numOfObject=libc.YoloDetectFromFilePy('../../test.jpg',network,90,600)
for i in range(0,numOfObject):
	print(libc.getClass(i))
	print(libc.getConf(i))
	print(libc.getX(i))
	print(libc.getY(i))
	print(libc.getW(i))
	print(libc.getH(i))
# YOLOv2_SpringEdition <img src="https://i.imgur.com/oYejfWp.png" title="Windows8" width="48">
###### YOLOv2 C++ library. (Train,Detect both)

* Remove pthread,opencv dependency.
* You need only 1 files for YOLO deep-learning.

<img src="https://i.imgur.com/ElCyyzT.png" title="Windows8" width="48"><img src="https://i.imgur.com/O5bye0l.png" width="48">



## Setup for train
You need `backup/`, `bin/`, `train/`, `obj.cfg`, `obj.data`, `obj.names`, `test.txt`, `train.txt`.


## Setup for detect

![](https://i.imgur.com/XjTlCMi.jpg)
```
Recall : 0.481481
Precision : 0.722222
Average detection time : 0.0363674
FPS : 27.4971
```
## Reference

## Technical issue

## Software requirement

## Hardware requirement



It has only 4 c-style function.

```cpp
void YoloTrain(char* _base_dir, char* _datafile, char* _cfgfile);
int* YoloLoad(char* cfgfile, char* weightsfile);
int YoloDetectFromFile(char* img_path, int* _net, float threshold, float* result, int result_sz);
int YoloDetectFromBytesImage(unsigned char* img, int w, int h, int* _net, float threshold, float* result, int result_sz);
```
You can use `install\YOLOv2SE.hpp`, `install\YOLOv2SE.cpp` in **c++**.

There is a example source code in `prj_example_*\`.

#### Requirement
* CUDA 8.0
* Visual Studio 2015

#### How to build.
1. open `network\yolo.weights.download.html`

Enjoy YOLO.

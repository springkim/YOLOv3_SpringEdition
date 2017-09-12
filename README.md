# YOLOv2_SpringEdition
#### YOLO C++ wrapper dll library

* remove pthread,opencv dependency.
* support multi-gpu
* c++ wrapper class

You need only 2 files for YOLO deep-learning.

`lnstall/YOLOv2SE.dll` , `install/cudnn43_5.dll`

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

# YOLOv2_SpringEdition <img src="https://i.imgur.com/oYejfWp.png" title="Windows8" width="48">
###### YOLOv2 C++ library. (Train,Detect both)

* Remove pthread,opencv dependency.
* You need only 1 files for YOLO deep-learning.

<img src="https://i.imgur.com/ElCyyzT.png" title="Windows8" width="48"><img src="https://i.imgur.com/O5bye0l.png" width="48">



## Setup for train
You need `backup/`, `bin/`, `train/`, `obj.cfg`, `obj.data`, `obj.names`, `test.txt`, `train.txt`.
All files are included in my repository. You can execute [YOLOv2/train.bat](https://github.com/springkim/YOLOv2_SpringEdition/blob/master/YOLOv2/train.bat) for training.

## Setup for detect
#### 1. Download pre-trained model and dll files.
[YOLOv2_SE_Detection_Example/download_pretrained_model_and_dll.bat](https://github.com/springkim/YOLOv2_SpringEdition/blob/master/YOLOv2_SE_Detection_Example/download_pretrained_model_and_dll.bat)

#### 2. Run

Example source is in my repository. You need only 1 header file and 2 dll files.


```
Recall : 0.481481
Precision : 0.722222
Average detection time : 0.0363674
FPS : 27.4971
```
![](https://i.imgur.com/XjTlCMi.jpg)
## Reference
The class `YOLOv2` that in `YOLOv2SE.h` has 3 method.
```cpp
void Create(std::string weights,std::string cfg,std::string names);
```
This method load trained model(**weights**), network configuration(**cfg**) and class naming file(**names**)
* **Parameter**
	* **weights** : trained model path(e.g. "obj.weights")
	* **cfg** : network configuration file(e.g. "obj.cfg")
	* **names** : class naming file(e.g. "obj.names")

```cpp
std::vector<BoxSE> Detect(cv::Mat img, float threshold);
std::vector<BoxSE> Detect(std::string file, float threshold);
std::vector<BoxSE> Detect(IplImage* img, float threshold);
```
This method is detecting objects of `file`,`cv::Mat` or `IplImage`.
* **Parameter**
	* **file** : image file path
	* **img** : 3-channel image.
	* **threshold** : It removes predictive boxes if there score is less than threshold.

```cpp
void Release();
```
Release loaded network.

## Technical issue

Original YOLOv2 has so many dependencies. I removed that.

A YOLOv2_Train_SE.exe is automatically choosing multi-gpu training. and select backup weights.

## Software requirement

* Visual Studio 2015
* CUDA 8.0

## Hardware requirement

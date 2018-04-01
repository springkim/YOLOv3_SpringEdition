## YOLOv2_SpringEdition <img src="https://i.imgur.com/oYejfWp.png" title="Windows8" width="48">

<img src="https://i.imgur.com/ElCyyzT.png" title="Windows8" width="48"><img src="https://i.imgur.com/O5bye0l.png" width="48"><img src="https://i.imgur.com/kmfOMZz.png" width="48"><img src="https://i.imgur.com/6OT8yM9.png" width="48">

#### YOLOv2 C++ Windows and Linux interface library. (Train,Detect both)

* Remove pthread,opencv dependency.
* You need only 1 files for YOLO deep-learning.
* Support windows, linux as same interface.

#### Do you want train YOLOv2 as double click? and detect using YOLOv2 as below?
```cpp
YOLOv3 detector;
detector.Create("coco.weights", "coco.cfg", "coco.names");
cv::Mat img=cv::imread("a.jpg");
std::vector<BoxSE> boxes = detector.Detect(img, 0.5F);
```
* Then you've come to the right place.

### 1. Setup for train.
You need only 2 files for train that are **YOLOv2SE_Train.exe** and **cudnn64_5.dll** on Windows.
If you are on Linux, then you need only **YOLOv2SE_Train**.
This files are in `YOLOv2_SpringEdition/bin`.

The requirement interface not changed. Same as **[pjreddie/darknet](https://github.com/pjreddie/darknet)**.

There is a example training directory `Yolov2_SpringEdition_Train/`. You can start training using above files.

Actually, all the interfaces are same with YOLOv1. So you can easily train your own data.

The **YOLOv2SE_Train.exe**'s arguments are [base directory],[data file path] and [cfg file path].

And YOLOv2SE_Train.exe is automatically choosing multi-gpu training. and select latest backup weights file.

### 2. Setup for detect

Just include **YOLOv2SE.h** and use it. See  `YOLOv2_SpringEdition_Test/`.

##### Reference

The class `YOLOv2` that in `YOLOv2SE.h` has 3 methods.
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



## Software requirement

* CMake
* CUDA 8.0(Maybe it works on CUDA9)
* OpenCV
* Visual Studio

## Hardware requirement

* NVIDIA GPU
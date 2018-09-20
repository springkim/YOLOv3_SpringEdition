YOLOv3_SpringEdition <img src="https://i.imgur.com/oYejfWp.png" title="Windows8" width="48">
--------------------------------------------------------------------------------------------


작성중....
VS2017
LINUX
기타 등등...

<img src="https://i.imgur.com/ElCyyzT.png" title="Windows8" width="48"><img src="https://i.imgur.com/O5bye0l.png" width="48"><img src="https://i.imgur.com/kmfOMZz.png" width="48"><img src="https://i.imgur.com/6OT8yM9.png" width="48">

#### YOLOv3 C++ Windows and Linux interface library. (Train,Detect both)

-	Remove pthread,opencv dependency.
-	You need only 1 files for YOLO deep-learning.
-	Support windows, linux as same interface.

#### Do you want train YOLOv3 as double click? and detect using YOLOv3 as below?

```cpp
YOLOv3 detector;
detector.Create("coco.weights", "coco.cfg", "coco.names");
cv::Mat img=cv::imread("a.jpg");
std::vector<BoxSE> boxes = detector.Detect(img, 0.5F);
```

-	Then you've come to the right place.

### 1. Setup for train.

#### 1.1. Train detector
You need only 2 files for train that are **YOLOv3SE_Train.exe** and **cudnn64_7.dll** on Windows. If you are on Linux, then you need only **YOLOv3SE_Train**. This files are in `YOLOv3_SpringEdition/bin`. or you can make it using `build_windows.bat` and `build_linux.sh`.

The requirement interface not changed. Same as **[pjreddie/darknet](https://github.com/pjreddie/darknet)**.

There is a example training directory `Yolov3_SpringEdition_Train/`. You can start training using above files.

Actually, all the interfaces are same with YOLOv2. So you can easily train your own data.

The **YOLOv3SE_Train.exe**'s arguments are `[option]`,`[base directory]`,`[data file path]` and `[cfg file path]`.

And YOLOv3SE_Train.exe is automatically choosing multi-gpu training. and select latest backup weights file.

Example : [Yolov3_SpringEdition_Train/DetectorExample/](Yolov3_SpringEdition_Train/DetectorExample/)

##### Sample directory structure with VOC2007 dataset.
```
┌ voc2007train
│  ├ 000012.jpg
│  ├ 000012.txt
│  ├ 000017.jpg
│  ...
│  └ 009961.txt
├ backup
├ yolov3_darknet53.cfg
├ voc2007.data
├ voc2007.names
├ train.txt
├ cudnn64_7.dll
└ YOLOv3SE_ Train.exe
```
##### Sample run argument with STL10 dataset.
```
"YOLOv3SE_Train.exe" detector . voc2007.data darknet53.cfg
```
#### 1.2. Train classifier
Example : [Yolov3_SpringEdition_Train/ClassifierExample/](Yolov3_SpringEdition_Train/ClassifierExample/)

##### Sample directory structure with STL10 dataset.
```
┌ stl10train
│  ├ airplane
│  ├ bird
│  ├ car
│  ...
│  └ truck
├ backup
├ resnext50.cfg
├ stl10.data
├ stl10.names
├ train.txt
├ cudnn64_7.dll
└ YOLOv3SE_ Train.exe
```
##### Sample run argument with STL10 dataset.
```
"YOLOv3SE_Train.exe" classifier . stl10.data resnext50.cfg
```

### 2. Setup for detect

**Do not change "batch" and "subdivisions" in cfg file. It automatically read those values as 1 when the network is on testing mode.**

Just include **YOLOv3SE.h** and use it. See `YOLOv3_SpringEdition_Test/`. You need only **YOLOv3SE.h**, **libYOLOv3SE.dll** and **cudnn64_7.dll** for detect.

###### 1. Go to [Yolov3_SpringEdition_Test](Yolov3_SpringEdition_Test)
###### 2. Run "download_cudnn64_7.dll.bat" if you're in Windows.
###### 3. Download "voc2007valid", "yolov3_darknet53.weights" for detection.
###### Download "stl10valid", "resnext50_256_10000.weights" for classification.
###### 4. Run VS solution or build.sh.

##### Reference

The class `YOLOv3` that in `YOLOv3SE.h` has 3 methods.

```cpp
void Create(std::string weights,std::string cfg,std::string names);
```

This method load trained model(**weights**), network configuration(**cfg**) and class naming file(**names**\)*

* **Parameter**
	* **weights** : trained model path(e.g. "obj.weights") 
	* **cfg** : network configuration file(e.g. "obj.cfg") 
	* **names** : class naming file(e.g. "obj.names")

```cpp
std::vector<BoxSE> Detect(cv::Mat img, float threshold);
std::vector<BoxSE> Detect(std::string file, float threshold);
std::vector<BoxSE> Detect(IplImage* img, float threshold);
int Classify(IplImage* img);
int Classify(cv::Mat img);
int Classify(std::string file);
```

This method is detecting objects or classify of `file`,`cv::Mat` or `IplImage`.
* **Parameter** 
	* **file** : image file path 
	* **img** : 3-channel image. 
	* **threshold** : It removes predictive boxes if there score is less than threshold.


```cpp
void Release();
```
Release loaded network.



Technical issue
---------------

Original YOLOv3(darknet) is linux version. And **[AlexeyAB](https://github.com/AlexeyAB/darknet)** already made YOLOv3 Windows version. But, his detection method is too slow on Windows. I don't know why exactly. Maybe it has bottleneck. So, I converted **[darknet](https://github.com/pjreddie/darknet)**(YOLOv3 only) again.

* YOLOv1 doesn't work.

change log
----------

**build_windows.bat** and **build_linux.sh** will download automatically correct version of cudnn. and build as cmake.

```
Windows + 1080ti + CUDA8.0 + cudnn7.1 + yolov3      = 36FPS
Windows + 1080ti + CUDA9.0 + cudnn7.1 + yolov3      = 36FPS
Windows + 1080   + CUDA9.0 + cudnn7.1 + yolov3      = 27FPS
Windows + 1080   + CUDA9.0 + cudnn7.1 + yolov3(spp) = 15FPS
Ubuntu  + 1080   + CUDA8.0 + cudnn7.1 + yolov3      = 30FPS
Ubuntu  + 1080   + CUDA9.0 + cudnn7.1 + yolov3      = 30FPS
```

Software requirement
--------------------

-	CMake
-	CUDA 8.0 or 9.0(9.1 is not working)
-	OpenCV(for testing)
-	Visual Studio 2015

Hardware requirement
--------------------

-	NVIDIA GPU

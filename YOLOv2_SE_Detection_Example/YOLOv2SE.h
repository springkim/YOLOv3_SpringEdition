
/*
*  YOLOv2_SE.h
*  YOLOv2_SpringEdition
*
*  Created by kimbom on 2017. 10. 6...
*  Copyright 2017 Sogang Univ. All rights reserved.
*
*/
#if !defined(YOLO_7E0_05_17_YOLO_H_INCLUDED)
#define YOLO_7E0_05_17_YOLO_H_INCLUDED
#include<fstream>
#include<string>
#include<vector>
#include<functional>
#include<exception>
#include<mutex>
#include<Windows.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#pragma warning(disable:4305)
#pragma warning(disable:4290)
#ifndef  SPRING_EDITION_BOX
#define SPRING_EDITION_BOX
/**
*	@brief 이 클래스는 cv::Rect를 확장한 것으로 클래스값과 스코어값이 추가되었습니다.
*	@author kimbomm
*	@date 2017-10-05
*
*	@see	https://github.com/springkim/FasterRCNN_SpringEdition
*	@see	https://github.com/springkim/YOLOv2_SpringEdition
*/
class BoxSE : public cv::Rect {
public:
	int m_class = -1;
	float m_score = 0.0F;
	std::string m_class_name;
	BoxSE() {
		m_class_name = "Unknown";
	}
	BoxSE(int c, float s, int _x, int _y, int _w, int _h, std::string name = "")
		:m_class(c), m_score(s) {
		this->x = _x;
		this->y = _y;
		this->width = _w;
		this->height = _h;
		char* lb[5] = { "th","st","nd","rd","th" };
		if (name.length() == 0) {
			m_class_name = std::to_string(m_class) + lb[m_class < 4 ? m_class : 4] + " class";
		}
	}
};
#endif
class YOLOv2 {
private:
	using YoloLoadType = int*(*)(char* cfg, char* weights);
	using YoloTrainType = void(*)(char* _base_dir, char* _datafile, char* _cfgfile);
	using YoloDetectFromFileType = int(*)(char* img_path, int* _net, float threshold, float* result, int result_sz);
	using YoloDetectFromIplImageType = int(*)(IplImage* img, int* _net, float threshold, float* result, int result_sz);
private:
	YoloLoadType YoloLoad = nullptr;
	YoloTrainType YoloTrain = nullptr;
	YoloDetectFromFileType YoloDetectFromFile = nullptr;
	YoloDetectFromIplImageType YoloDetectFromIplImage = nullptr;
protected:
	int* m_network = nullptr;
	HMODULE m_hmod = nullptr;
	std::vector<std::string> m_names;
public:
	void Create(std::string weights,std::string cfg,std::string names) {
		//Load
		this->m_network = YoloLoad(const_cast<char*>(cfg.c_str()), const_cast<char*>(weights.c_str()));
		if (names.length() > 0) {
			std::fstream fin(names, std::ios::in);
			if (fin.is_open() == true) {
				this->m_names.clear();
				while (fin.eof() == false) {
					std::string str;
					std::getline(fin, str);
					if (str.length() > 0) {
						this->m_names.push_back(str);
					}
				}
				fin.close();
			}
		}
	}
	void Release() {
		if (this->m_hmod != nullptr) {
			FreeLibrary(this->m_hmod);
			m_hmod = nullptr;
		}
	}
	std::vector<BoxSE> Detect(cv::Mat img, float threshold) {
		IplImage* iplimg = new IplImage(img);
		std::vector<BoxSE> boxes= this->Detect(iplimg, threshold);
		delete iplimg;
		return boxes;
	}
	std::vector<BoxSE> Detect(std::string file, float threshold) {
		float* result = new float[6000];
		int n = YoloDetectFromFile(const_cast<char*>(file.c_str()), this->m_network, threshold, result, 6000);
		std::vector<BoxSE> boxes;
		for (int i = 0; i < n; i++) {
			BoxSE box;
			box.m_class = static_cast<int>(result[i * 6 + 0]);
			box.m_score = result[i * 6 + 1];
			box.x = static_cast<int>(result[i * 6 + 2]);
			box.y = static_cast<int>(result[i * 6 + 3]);
			box.width = static_cast<int>(result[i * 6 + 4]);
			box.height = static_cast<int>(result[i * 6 + 5]);
			if (this->m_names.size() > 0) {
				box.m_class_name = this->m_names[box.m_class];
			}
			boxes.push_back(box);
		}
		delete[] result;
		std::sort(boxes.begin(), boxes.end(), [](BoxSE a, BoxSE b)->bool { return a.m_score > b.m_score; });
		return boxes;
	}
	std::vector<BoxSE> Detect(IplImage* img, float threshold) {
		float* result = new float[600];
		int n = YoloDetectFromIplImage(img,this->m_network, threshold, result, 600);
		std::vector<BoxSE> boxes;
		for (int i = 0; i < n; i++) {
			BoxSE box;
			box.m_class = static_cast<int>(result[i * 6 + 0]);
			box.m_score = result[i * 6 + 1];
			box.x = static_cast<int>(result[i * 6 + 2]);
			box.y = static_cast<int>(result[i * 6 + 3]);
			box.width = static_cast<int>(result[i * 6 + 4]);
			box.height = static_cast<int>(result[i * 6 + 5]);
			if (this->m_names.size() > 0) {
				box.m_class_name = this->m_names[box.m_class];
			}
			boxes.push_back(box);
		}
		delete[] result;
		std::sort(boxes.begin(), boxes.end(), [](BoxSE a, BoxSE b)->bool { return a.m_score > b.m_score; });
		return boxes;
	}

	YOLOv2() {
		std::string dll = "YOLOv2_SE.dll";
		m_hmod = LoadLibraryA(dll.c_str());
		if (m_hmod == nullptr) {
			::MessageBoxA(NULL, "YOLOv2_SE.dll not found. or can't load dependency dlls", "Fatal", MB_OK);
			exit(1);
		}
		YoloLoad = (YoloLoadType)GetProcAddress(m_hmod, "YoloLoad");
		YoloTrain = (YoloTrainType)GetProcAddress(m_hmod, "YoloTrain");
		YoloDetectFromFile = (YoloDetectFromFileType)GetProcAddress(m_hmod, "YoloDetectFromFile");
		YoloDetectFromIplImage = (YoloDetectFromIplImageType)GetProcAddress(m_hmod, "YoloDetectFromIplImage");
	}
	~YOLOv2() {
		this->Release();
	}
};

#endif
#if !defined(YOLO_7E0_05_17_YOLO_H_INCLUDED)
#define YOLO_7E0_05_17_YOLO_H_INCLUDED
///c++ header
#include<fstream>
#include<string>
#include<vector>
#include<functional>
#include<exception>
#include<mutex>
///windows header
#include<Windows.h>
///opencv header
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#pragma warning(disable:4305)
#pragma warning(disable:4290)
namespace yolo {
	class YoloBox: public cv::Rect_<float>{
	public:
		int m_class = -1;
		float m_confidence_value = 0.0F;
		std::string m_name;
	public:
		bool operator<(const YoloBox& other) const;
		bool operator>(const YoloBox& other)const;
	};
	class YoloDetector {
	private:
		using YoloLoadType = int*(*)(char* cfg, char* weights);
		using YoloTrainType = void(*)(char* _base_dir, char* _datafile, char* _cfgfile);
		using YoloDetectFromFileType = int(*)(char* img_path, int* _net, float threshold, float* result, int result_sz);
		//using YoloDetectFromCvMatType = int(*)(cv::Mat img, int* _net, float threshold, float* result, int result_sz);
		using YoloDetectFromBytesImageType = int(*)(uchar* img, int w,int h,int* _net, float threshold, float* result, int result_sz);
	private:
		YoloLoadType YoloLoad = nullptr;
		YoloTrainType YoloTrain = nullptr;
		YoloDetectFromFileType YoloDetectFromFile = nullptr;
		//YoloDetectFromCvMatType YoloDetectFromCvMat = nullptr;
		YoloDetectFromBytesImageType YoloDetectFromBytesImage = nullptr;
	protected:
		int* m_network = nullptr;
		HMODULE m_hmod = nullptr;
		std::vector<std::string> m_names;
	public:
		void Create() throw(std::exception);
		void Release();
		void Load(std::string cfg, std::string weights, std::string names = "");
		std::vector<YoloBox> Detect(cv::Mat img, float threshold);
		std::vector<YoloBox> Detect(std::string file, float threshold);
		std::vector<YoloBox> Detect(uchar* bytes_image,int w,int h, float threshold);
		cv::Mat Draw(cv::Mat _img, std::vector<YoloBox> boxes);
		void Train(std::string base_dir, std::string data, std::string cfg);
		YoloDetector(bool auto_create = true);
		~YoloDetector();
	};
}
#endif
// dllmain.cpp: DLL 응용 프로그램의 진입점을 정의합니다.
#include <Windows.h>
#include<string>
#include<iostream>
#include"ispring/All.h"	//@see https://github.com/springkim/ISpring
#include"YOLOv2SE.h"
std::vector<BoxSE> GetGroundTruth(std::string file) {
	std::vector<BoxSE> ret;
	std::string tag = file.substr(0, file.find_last_of(".")) + ".tag";
	cv::Mat img = cv::imread(file);
	std::ifstream fin;
	fin.open(tag, std::ios::in);
	int n;
	fin >> n;
	while (n--) {
		BoxSE box;
		fin >> box.m_class;
		float x1, y1, x2, y2;
		fin >> x1 >> y1 >> x2 >> y2;
		box.x = static_cast<int>(x1*img.cols);
		box.y = static_cast<int>(y1*img.rows);
		box.width = static_cast<int>((x2 - x1)*img.cols);
		box.height = static_cast<int>((y2 - y1)*img.rows);
		box.m_score = -1;
		ret.push_back(box);
	}
	fin.close();
	return ret;
}
int main() {

	std::string weights = "obj_final.weights";
	std::string cfg = "obj.cfg";
	std::string names = "obj.names";
	YOLOv2 detector;
	detector.Create(weights, cfg, names);

	std::vector<std::string> files = ispring::File::FileList("testImages", "*.jpg");
	std::vector<cv::Mat> results;
	std::pair<float, float> RP = { 0.0F,0.0F };
	for (auto& file : files) {
		ispring::Timer::Tick("detect");
		std::vector<BoxSE> boxes = detector.Detect(file, 0.1F);
		ispring::Timer::Tock("detect");

		std::vector<BoxSE> ground_truth = GetGroundTruth(file);
		std::pair<float, float> lRP = ispring::CVEval::GetRecallPrecisionSingleClass(ground_truth, boxes);
		RP.first += lRP.first;
		RP.second += lRP.second;
		
		cv::Mat img = cv::imread(file);
		for (auto&box : boxes) {
			ispring::CVEval::DrawBoxSE(img, box,cv::Scalar(0,0,255));
		}
		for (auto&box : ground_truth) {
			ispring::CVEval::DrawBoxSE(img, box,cv::Scalar(0,255,0));
		}
		results.push_back(img);
	}
	detector.Release();
	cv::Mat img = ispring::CV::GlueImage(results);
	ispring::CV::DisplayImage2(img, true);
	cv::imwrite("result.jpg", img);
	double T = ispring::Timer::Watch("detect").avg;
	std::cout << "Recall : " << RP.first / files.size() << std::endl;
	std::cout << "Precision : " << RP.second / files.size() << std::endl;
	std::cout << "Average detection time : " << T << std::endl;
	std::cout << "FPS : " << 1 / T << std::endl;
	return 0;
}
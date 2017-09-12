#ifndef _WINDLL
#include"stdafx.h"
#include<string>
#include"install/YOLOv2SE.hpp"

int main() {
	std::string file = "../../test.jpg";
	yolo::YoloDetector yd;
	yd.Load("../../network/yolo.cfg", "../../network/yolo.weights", "../../network/yolo.names");

	cv::Mat img = cv::imread(file);
	std::vector<yolo::YoloBox> boxes = yd.Detect(img, 0.9F);

	cv::Mat draw = yd.Draw(img, boxes);
	cv::imshow("img", draw);
	cv::waitKey();
	cv::destroyAllWindows();
}
#endif
#include"YOLOv3SE.h"
#include<chrono>
#include<iostream>
void Yolo3Test(std::string weights, std::string cfg,std::string names) {
	YOLOv3 detector;
	detector.Create(weights,cfg,names);
	std::vector<cv::Scalar> colors;
	for (int i = 0; i < 80; i++) {
		colors.push_back(cv::Scalar(rand() % 127 + 128, rand() % 127 + 128, rand() % 127 + 128));
	}
	cv::VideoCapture vc("D:/01.KEPCO/train_data/05.excavator/¹ÌÅÂ±ë/youtube2/Real Life Rocket League  Car Soccer Game.mp4");
	std::fstream fin("voc2007valid.txt", std::ios::in);
	cv::Mat frame;
	while(vc.read(frame)){
	//while (fin.eof() == false) {
		//std::string line;
		//std::getline(fin, line);
		//if (line.length() == 0)break;
		//cv::Mat img = cv::imread(line);
		cv::Mat img = cv::imread("C:/Users/VIRNECT/Desktop/KakaoTalk_Photo_20190321_1730_56891.jpg");
		std::chrono::system_clock::time_point t_beg, t_end;
		std::chrono::duration<double> diff;
		t_beg = std::chrono::system_clock::now();
		auto boxes = detector.Detect(img, 0.3F);
		t_end = std::chrono::system_clock::now();
		diff = t_end - t_beg;
		std::cout << "FPS : " << 1.0 / diff.count() << "\t" << diff.count() << std::endl;
		//continue;
		for (auto&box : boxes) {
			cv::putText(img, detector.Names(box.m_class), box.tl()-cv::Point(0,5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(160,96,173), 2);
			cv::rectangle(img, box, cv::Scalar(160, 96, 173), 2);
		}
		cv::imwrite("original.jpg", img);
		if (cv::waitKey(0) == 27) {
			break;
		}
		break;
	}
	cv::destroyAllWindows();
	detector.Release();
}
void ClassifyTest() {
	YOLOv3 classifier;
	classifier.Create("resnext50_256_10000.weights", "resnext50.cfg", "stl10.names");

	std::fstream fin("stl10valid.txt", std::ios::in);
	int total = 0;
	int correct = 0;
	while (true) {
		std::string line;
		std::getline(fin, line);
		if (line.length() == 0)break;
		int r = classifier.Classify(line);
		if (line.find(classifier.Names(r)) != std::string::npos) {
			correct++;
		}
		total++;
		std::cout << "accuracy : " << (float)correct / total << "(" << correct << "/" << total << ")" << std::endl;
	}
	classifier.Release();
	//http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html 
	//5000, 0.613
	//10000, 0.7685
}
int main() {
	///Test YOLOv3 based on darknet53
	//Yolo3Test("yolov3_darknet53.weights","yolov3_darknet53_coco.cfg","coco.names");

	///Test YOLOv3 based on darknet53 with SPP(spatial pyramid pooling)
	//Yolo3Test("yolov3_darknet53.weights", "yolov3_darknet53_coco.cfg", "coco.names");
	Yolo3Test("yolov3_darknet53_spp.weights", "yolov3_darknet53_spp_coco.cfg", "coco.names");

	///Test Classifier
	//ClassifyTest();
	return 0;
}

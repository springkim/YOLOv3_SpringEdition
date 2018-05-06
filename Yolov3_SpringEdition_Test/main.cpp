#include"YOLOv3SE.h"
#include<chrono>
void Yolo3Test() {
	YOLOv3 detector;
	detector.Create("darknet53.weights", "darknet53_coco.cfg", "coco.names");
	
	std::vector<cv::Scalar> colors;
	for (int i = 0; i < 80; i++) {
		colors.push_back(cv::Scalar(rand() % 127 + 128, rand() % 127 + 128, rand() % 127 + 128));
	}
	
	std::fstream fin("valid.txt", std::ios::in);
	while (fin.eof() == false) {
		std::string line;
		std::getline(fin, line);
		if (line.length() == 0)break;
		cv::Mat img = cv::imread(line);
		
		std::chrono::system_clock::time_point t_beg, t_end;
		std::chrono::duration<double> diff;
		t_beg = std::chrono::system_clock::now();
		auto boxes = detector.Detect(img, 0.5F);
		t_end = std::chrono::system_clock::now();
		diff = t_end - t_beg;
		std::cout << "FPS : " << 1.0 / diff.count() << "\t" << diff.count() << std::endl;
		continue;
		for (auto&box : boxes) {
			cv::putText(img, detector.Names(box.m_class), box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, colors[box.m_class], 2);
			cv::rectangle(img, box, colors[box.m_class], 2);
		}
		cv::imshow("original", img);
			

		if (cv::waitKey(0) == 27) {
			break;
		}
	}
	cv::destroyAllWindows();
}
int main() {
	Yolo3Test();

	return 0;
}

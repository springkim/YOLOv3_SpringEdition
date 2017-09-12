#include "YOLOv2SE.hpp"

bool yolo::YoloBox::operator<(const yolo::YoloBox& other)const {
	return this->m_confidence_value < other.m_confidence_value;
}
bool yolo::YoloBox::operator>(const yolo::YoloBox& other)const {
	return this->m_confidence_value > other.m_confidence_value;
}
void yolo::YoloDetector::Create() throw(std::exception) {

	std::string dll;
#ifdef _DEBUG
	dll = "YOLOv2SEd.dll";
#else
	dll = "YOLOv2SE.dll";
#endif
	m_hmod = LoadLibraryA(dll.c_str());
	if (m_hmod == nullptr) {
		throw std::exception(std::string(dll + " not found").c_str());
	}
	YoloLoad = (YoloLoadType)GetProcAddress(m_hmod, "YoloLoad");
	YoloTrain = (YoloTrainType)GetProcAddress(m_hmod, "YoloTrain");
	YoloDetectFromFile = (YoloDetectFromFileType)GetProcAddress(m_hmod, "YoloDetectFromFile");
	//YoloDetectFromCvMat = (YoloDetectFromCvMatType)GetProcAddress(m_hmod, "YoloDetectFromCvMat");
	YoloDetectFromBytesImage = (YoloDetectFromBytesImageType)GetProcAddress(m_hmod, "YoloDetectFromBytesImage");
}
void yolo::YoloDetector::Release() {
	if (this->m_hmod != nullptr) {
		FreeLibrary(this->m_hmod);
	}
}
void yolo::YoloDetector::Load(std::string cfg, std::string weights, std::string names/* = ""*/) {
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
std::vector<yolo::YoloBox> yolo::YoloDetector::Detect(cv::Mat img, float threshold) {
	float* result = new float[600];
	int n = YoloDetectFromBytesImage(img.data,img.cols,img.rows, this->m_network, threshold, result, 600);
	std::vector<YoloBox> boxes;
	for (int i = 0; i < n; i++) {
		YoloBox box;
		box.m_class = static_cast<int>(result[i * 6 + 0]);
		box.m_confidence_value = result[i * 6 + 1];
		box.x = result[i * 6 + 2];
		box.y = result[i * 6 + 3];
		box.width = result[i * 6 + 4];
		box.height = result[i * 6 + 5];
		if (this->m_names.size() > 0) {
			box.m_name = this->m_names[box.m_class];
		}
		boxes.push_back(box);
	}
	delete[] result;
	std::sort(boxes.begin(), boxes.end(), std::greater<YoloBox>());
	return boxes;
}
std::vector<yolo::YoloBox> yolo::YoloDetector::Detect(std::string file, float threshold) {
	return this->Detect(cv::imread(file), threshold);
}
std::vector<yolo::YoloBox> yolo::YoloDetector::Detect(uchar* bytes_image,int w,int h, float threshold) {
	return this->Detect(cv::Mat(h, w, CV_8UC3, bytes_image), threshold);
}
void yolo::YoloDetector::Train(std::string base_dir, std::string data, std::string cfg) {
	this->YoloTrain(const_cast<char*>(base_dir.c_str())
					, const_cast<char*>(data.c_str())
					, const_cast<char*>(cfg.c_str()));
}
cv::Mat yolo::YoloDetector::Draw(cv::Mat _img, std::vector<yolo::YoloBox> boxes) {
	cv::Mat img = _img.clone();
	int longer = std::max(img.cols, img.rows);
	cv::Scalar color(0, 0, 255); 
	cv::Scalar color_text(255, 255, 255);
	int font = cv::FONT_HERSHEY_SIMPLEX;
	double fontScale = 0.7;
	int thickness = 1;
	int baseline = 0;
	for (auto& box : boxes) {
		cv::Size text = cv::getTextSize(box.m_name,  font,fontScale, thickness, &baseline);
		cv::rectangle(img, cv::Point((int)box.x, (int)box.y), cv::Point((int)box.x, (int)box.y) +cv::Point(text.width,-text.height-baseline-baseline), color, CV_FILLED);
		cv::putText(img, box.m_name, cv::Point((int)box.x, (int)box.y) + cv::Point(0, -baseline), font, fontScale, color_text, thickness);

		cv::rectangle(img, box, color, thickness);
	}
	return img;
}
yolo::YoloDetector::YoloDetector(bool auto_create/*=true*/) {
	if (auto_create == true) {
		this->Create();
	}
}
yolo::YoloDetector::~YoloDetector() {
	this->Release();
}
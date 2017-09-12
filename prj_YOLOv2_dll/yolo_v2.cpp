#include"yolo_v2.h"
int YoloDetect(image img, int* _net, float threshold, float* result, int result_sz) {
	network* net = (network*)_net;
	image im = img;
	image sized = resize_image(im, net->w, net->h);
	
	layer l = net->layers[net->n - 1];
	box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
	float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
	for (int j = 0; j < l.w*l.h*l.n; ++j) {
		probs[j] = (float*)calloc(l.classes, sizeof(float));
	}
	
	network_predict(*net, sized.data);
	get_region_boxes(l, 1, 1, threshold, probs, boxes, 0, 0);
	do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, 0.4F);
	struct {
		float m_class;
		float m_confidence_value;
		float m_x, m_y, m_w, m_h;
	}yolobox;
	std::vector<decltype(yolobox)> vec_box;
	for (int i = 0; i < l.w*l.h*l.n; ++i) {
		int _class = max_index(probs[i], l.classes);
		float comfidence_value = probs[i][_class];
		if (comfidence_value >= threshold) {
			const box& b = boxes[i];
			float left = (b.x - b.w / 2.0F)*im.w; if (left < 0) left = 0.0F;
			float right = (b.x + b.w / 2.0F)*im.w; if (right > im.w - 1) right = im.w - 1;
			float top = (b.y - b.h / 2.0F)*im.h; if (top < 0) top = 0;
			float bot = (b.y + b.h / 2.0F)*im.h; if (bot > im.h - 1) bot = im.h - 1;
			yolobox.m_class = _class;
			yolobox.m_confidence_value = comfidence_value;
			yolobox.m_x = left;
			yolobox.m_y = top;
			yolobox.m_w = fabs(right - left);
			yolobox.m_h = fabs(bot - top);
			vec_box.push_back(yolobox);
		}
	}
	//prob이 높은순으로 정렬
	std::stable_sort(std::begin(vec_box), std::end(vec_box)
					 , [](const decltype(yolobox)& a, const decltype(yolobox)& b)->bool {
		return a.m_confidence_value > b.m_confidence_value;
	});
	size_t i = 0;
	for (i = 0; i < vec_box.size(); i++) {
		if (i * 6 + 5 >= result_sz) {
			break;
		}
		result[i * 6 + 0] = vec_box[i].m_class;
		result[i * 6 + 1] = vec_box[i].m_confidence_value;
		result[i * 6 + 2] = vec_box[i].m_x;
		result[i * 6 + 3] = vec_box[i].m_y;
		result[i * 6 + 4] = vec_box[i].m_w;
		result[i * 6 + 5] = vec_box[i].m_h;
	}
	free_image(im);
	free_image(sized);
	free(boxes);
	free_ptrs((void **)probs, l.w*l.h*l.n);
	return (int)i;
}
extern "C" {
	/*
	*	@YoloTrain
	*	@param1 : 기준 디렉터리
	*	@param2: data파일 이름(파일만 명시)
	*	@param3 : cfg파일 이름(파일만 명시)
	*/
	__declspec(dllexport) void YoloTrain(char* _base_dir, char* _datafile, char* _cfgfile) {
		std::string base_dir(_base_dir);
		std::string datafile(_datafile);
		std::string cfgfile(_cfgfile);
		//모든 파일들은 base_dir을 기준으로 찾습니다.(상대 경로일경우 절대경로로 바꿔줍니다)
		char rpath[_MAX_PATH];
		strcpy(rpath, base_dir.c_str());
		char apath[_MAX_PATH];
		_fullpath(apath, rpath, _MAX_PATH);
		base_dir.assign(apath, apath + strlen(apath));
		//현재 실행파일의 경로를 저장합니다.
		char curr_dir[_MAX_PATH];
		GetCurrentDirectoryA(_MAX_PATH, curr_dir);
		_chdir(base_dir.c_str());
		//콘솔응용프로그램이 아니거나, dll프로젝트일경우 콘솔창을 생성합니다.
#if !defined(_CONSOLE) || defined(_WINDLL)
		AllocConsole();
		freopen("CONOUT$", "w", stdout);
#endif
		int gpu_num = 0;
		cudaGetDeviceCount(&gpu_num);
		if (gpu_num == 0) {
			std::cout << "No GPUs" << std::endl;
			return;
		}
		int* gpu_indexes = new int[gpu_num];
		for (int i = 0; i < gpu_num; i++) {
			gpu_indexes[i] = i;
			cudaDeviceProp cdp;
			cudaGetDeviceProperties(&cdp, i);
			std::cout << i << " : " << cdp.name << std::endl;
		}
		std::cout << "Train start!!" << std::endl;
		train_detector(
			const_cast<char*>(datafile.c_str())
			, const_cast<char*>(cfgfile.c_str())
			, nullptr
			, gpu_indexes
			, gpu_num	/*number of gpu*/
			, 0);
#if !defined(_CONSOLE) || defined(_WINDLL)
		FreeConsole();
#endif
		_chdir(curr_dir);
	}
	/*
	*	@YoloLoad
	*	@param1 : cfg 파일 경로
	*	@param2: weights 파일 경로
	*	@comment : C++의 경우는 int*로 network파일을 담아두면 됩니다.
	*				, C#의 경우 IntPtr의 자료형으로 받으면 됩니다.
	*	@return : network class pointer
	*/
	using IntPtr = int*;
	__declspec(dllexport) int* YoloLoad(char* cfgfile, char* weightsfile) {
		network* net = new network[sizeof(network)];
		*net = parse_network_cfg(cfgfile);
		load_weights(net, weightsfile);
		set_batch_network(net, 1);
		srand(921126);
		return (int*)net;
	}
	/*
	*	@YoloDetect
	*	@param1 : image 파일 경로
	*	@param2: network
	*	@param3: threshold of confidence value
	*	@param4: float array
	*	@param5 : size of array
	*	@comment : result is looks like [class,confidence,x,y,w,h][class,confidence,x,y,w,h]...
	*	@return : number of detect
	====get rect example
	int sz=YoloDetect(path, network, threshold, result,6*100);
	-> it picks 100 boxes ordered by confidence value
	for (int i = 0; i < sz; i ++) {
	int kind = result[i * 6 + 0];
	int cval = result[i * 6 + 1];
	int left = result[i * 6 + 2];
	int top = result[i * 6 + 3];
	int width = result[i * 6 + 4];
	int height = result[i * 6 + 5];
	}
	====
	*/

	__declspec(dllexport) int YoloDetectFromFile(char* img_path, int* _net, float threshold, float* result, int result_sz) {
			image im = load_image_color(img_path, 0, 0);
			return YoloDetect(im, _net, threshold, result, result_sz);
	}
	/*__declspec(dllexport) int YoloDetectFromCvMat(cv::Mat img, int* _net, float threshold, float* result, int result_sz) {
		auto Mat2image = [](cv::Mat img)->image {
			IplImage src(img);
			unsigned char *data = (unsigned char *)src.imageData;
			image out = make_image(src.width, src.height, src.nChannels);
			int i, j, k, count = 0;;
			for (k = 0; k < src.nChannels; ++k)for (i = 0; i < src.height; ++i)for (j = 0; j < src.width; ++j)
				out.data[count++] = data[i*src.widthStep + j* src.nChannels + k] / 255.;
			rgbgr_image(out);
			return out;
		};
		image im = Mat2image(img);
		return YoloDetect(im, _net, threshold, result, result_sz);
	}*/
	__declspec(dllexport) int YoloDetectFromBytesImage(unsigned char* img, int w, int h, int* _net, float threshold, float* result, int result_sz) {
		auto Bytes2Mat = [](byte* img, int w, int h)->cv::Mat {
			return cv::Mat(h, w, CV_8UC3, img);
		};
		auto Mat2image = [](cv::Mat img)->image {
			IplImage src(img);
			unsigned char *data = (unsigned char *)src.imageData;
			image out = make_image(src.width, src.height, src.nChannels);
			int i, j, k, count = 0;;
			for (k = 0; k < src.nChannels; ++k)for (i = 0; i < src.height; ++i)for (j = 0; j < src.width; ++j)
				out.data[count++] = data[i*src.widthStep + j* src.nChannels + k] / 255.;
			rgbgr_image(out);
			return out;
		};
		image im = Mat2image(Bytes2Mat(img, w, h));
		return YoloDetect(im, _net, threshold, result, result_sz);
	}

	__declspec(dllexport) void YoloTrainPy(wchar_t* _base_dir, wchar_t* _datafile, wchar_t* _cfgfile) {
		std::wstring base_dir(_base_dir);
		std::wstring datafile(_datafile);
		std::wstring cfgfile(_cfgfile);
		YoloTrain(const_cast<char*>(std::string(base_dir.begin(), base_dir.end()).c_str())
				  , const_cast<char*>(std::string(datafile.begin(), datafile.end()).c_str())
				  , const_cast<char*>(std::string(cfgfile.begin(), cfgfile.end()).c_str()));
	}
	__declspec(dllexport) int __stdcall YoloSizePy() {
		return sizeof(network);
	}
	__declspec(dllexport)  void  __stdcall YoloLoadPy(wchar_t* _cfgfile, wchar_t* _weightsfile,int* _network) {
		network* net = (network*)_network;
		std::wstring w_cfgfile(_cfgfile);
		std::wstring w_weightsfile(_weightsfile);
		std::string cfgfile(w_cfgfile.begin(), w_cfgfile.end());
		std::string weightsfile(w_weightsfile.begin(), w_weightsfile.end());

		*net = parse_network_cfg(const_cast<char*>(cfgfile.c_str()));
		load_weights(net, const_cast<char*>(weightsfile.c_str()));
		set_batch_network(net, 1);
		srand(921126);
	}

	static float* resultPy = nullptr;
	__declspec(dllexport) int __stdcall YoloDetectFromFilePy(wchar_t* _img_path, int* _net, int _threshold, int result_sz) {
		std::wstring img_path(_img_path);
		std::string img_pathA(img_path.begin(), img_path.end());
		if (resultPy != nullptr) {
			delete[] resultPy;
		}
		resultPy = new float[result_sz];
		float threshold = _threshold / 100.0F;
		image im = load_image_color(const_cast<char*>(img_pathA.c_str()), 0, 0);
		int ret = YoloDetect(im, _net, threshold, resultPy, result_sz);
		//int ret= YoloDetectFromFile(const_cast<char*>(img_pathA.c_str()), _net, threshold, resultPy, result_sz);
		
		/*cv::Mat img = cv::imread(img_pathA);
		for (int i = 0; i < ret; i++) {
			int _class = resultPy[i * 6 + 0];
			int conf = resultPy[i * 6 + 1];
			int x = resultPy[i * 6 + 2];
			int y = resultPy[i * 6 + 3];
			int w = resultPy[i * 6 + 4];
			int h = resultPy[i * 6 + 5];
			cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2);
		}
		cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
		cv::imshow("img", img);
		cv::waitKey();
		cv::destroyAllWindows();*/

		return ret;
	}
	__declspec(dllexport) int  getClass(int index) {
		return  resultPy[index * 6 + 0];
	}
	__declspec(dllexport) int  getConf(int index) {
		return  resultPy[index * 6 + 1] * 100;
	}
	__declspec(dllexport) int  getX(int index) {
		return  resultPy[index * 6 + 2];
	}
	__declspec(dllexport) int  getY(int index) {
		return  resultPy[index * 6 + 3];
	}
	__declspec(dllexport) int  getW(int index) {
		return  resultPy[index * 6 + 4];
	}
	__declspec(dllexport) int  getH(int index) {
		return  resultPy[index * 6 + 5];
	}
}
#ifdef _WINDLL
#ifdef _DEBUG
class YoloV2_Debug_Printer {
public:
	YoloV2_Debug_Printer() {
		static bool debug_init = true;
		if (debug_init == true) {
			debug_init = false;
			std::cout << "****** YOLOv2_SpringEdition : version 2.0.0 ******" << "\n\n";
		}
	}
};
__declspec(selectany) YoloV2_Debug_Printer ydp;
#endif
#endif
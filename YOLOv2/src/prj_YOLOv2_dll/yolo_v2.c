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
	int result_idx = 0;
	for (int i = 0; i < l.w*l.h*l.n; ++i) {
		int _class = max_index(probs[i], l.classes);
		float confidence_value = probs[i][_class];
		if (confidence_value >= threshold) {
			const box b = boxes[i];
			float left = (b.x - b.w / 2.0F)*im.w; if (left < 0) left = 0.0F;
			float right = (b.x + b.w / 2.0F)*im.w; if (right > im.w - 1) right = im.w - 1;
			float top = (b.y - b.h / 2.0F)*im.h; if (top < 0) top = 0;
			float bot = (b.y + b.h / 2.0F)*im.h; if (bot > im.h - 1) bot = im.h - 1;

			if (result_idx * 6 + 5 < result_sz) {
				result[result_idx * 6 + 0] = _class;
				result[result_idx * 6 + 1] = confidence_value;
				result[result_idx * 6 + 2] = left;
				result[result_idx * 6 + 3] = top;
				result[result_idx * 6 + 4] = fabs(right - left);
				result[result_idx * 6 + 5] = fabs(bot - top);
				result_idx++;
			}
		}
	}
	
	free_image(im);
	free_image(sized);
	free(boxes);
	free_ptrs((void **)probs, l.w*l.h*l.n);
	return (int)result_idx;
}

/*
*	@YoloTrain
*	@param1 : 기준 디렉터리
*	@param2: data파일 이름(파일만 명시)
*	@param3 : cfg파일 이름(파일만 명시)
*/
__declspec(dllexport) void YoloTrain(char* _base_dir, char* _datafile, char* _cfgfile) {
	//모든 파일들은 base_dir을 기준으로 찾습니다.(상대 경로일경우 절대경로로 바꿔줍니다)
	char rpath[_MAX_PATH];
	strcpy(rpath, _base_dir);
	char apath[_MAX_PATH];
	_fullpath(apath, rpath, _MAX_PATH);
	
	//현재 실행파일의 경로를 저장합니다.
	char curr_dir[_MAX_PATH];
	GetCurrentDirectoryA(_MAX_PATH, curr_dir);
	_chdir(apath);
	//콘솔응용프로그램이 아니거나, dll프로젝트일경우 콘솔창을 생성합니다.
#if !defined(_CONSOLE) || defined(_WINDLL)
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
#endif
	int gpu_num = 0;
	cudaGetDeviceCount(&gpu_num);
	if (gpu_num == 0) {
		puts("No Gpus");
		return;
	}
	int* gpu_indexes = (int*)calloc(gpu_num, sizeof(int));
	for (int i = 0; i < gpu_num; i++) {
		gpu_indexes[i] = i;
		struct cudaDeviceProp cdp;
		cudaGetDeviceProperties(&cdp, i);
		
		printf("%d : %s\n", i, cdp.name);
	}
	char name[MAX_PATH];
	char* backup = NULL;
	for (int i = 44000; i >= 100;) {
		sprintf(name, "backup/obj_%d.weights", i);
		if (PathFileExistsA(name)==TRUE) {
			printf("%s is lastest backup file",name);
			backup = name;
			break;
		}
		if (i <= 1000) {
			i -= 100;
		} else {
			i -= 1000;
		}
	}
	puts("Train start!!");
	train_detector(
		_datafile
		, _cfgfile
		, backup
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

__declspec(dllexport) int* YoloLoad(char* cfgfile, char* weightsfile) {
	network* net = (network*)malloc(sizeof(network));
	memset(net, 0, sizeof(network));
	*net = parse_network_cfg(cfgfile);
	load_weights(net, weightsfile);
	set_batch_network(net, 1);
	srand(920217);
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
extern  image ipl_to_image(IplImage* src);
__declspec(dllexport) int YoloDetectFromIplImage(IplImage* src, int* _net, float threshold, float* result, int result_sz) {
	image im = ipl_to_image(src);
	rgbgr_image(im);
	return YoloDetect(im, _net, threshold, result, result_sz);
}

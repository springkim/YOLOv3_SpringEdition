#include"yolo_v2.h"

int YoloDetect(image img, int* _net, float threshold, float* result, int result_sz) {
	network* net = (network*)_net;
	image im = img;

	image sized = letterbox_image(im, net->w, net->h);
	layer l = net->layers[net->n - 1];
	int j;
	box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
	float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

	network_predict(net, sized.data);
	get_region_boxes(l, im.w, im.h, net->w, net->h, threshold, probs, boxes, NULL, 0, 0, 0.5F, 1);
	do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, 0.45F);
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
	free_image(sized);
	free(boxes);
	free_ptrs((void **)probs, l.w*l.h*l.n);
	return (int)result_idx;
}

/*
*	@YoloLoad
*	@param1 : cfg 파일 경로
*	@param2: weights 파일 경로
*	@comment : C++의 경우는 int*로 network파일을 담아두면 됩니다.
*				, C#의 경우 IntPtr의 자료형으로 받으면 됩니다.
*	@return : network class pointer
*/

DLL_MACRO int* YoloLoad(char* cfgfile, char* weightsfile) {
	network* net = (network*)malloc(sizeof(network));
	memset(net, 0, sizeof(network));
	net = parse_network_cfg(cfgfile);
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

DLL_MACRO int YoloDetectFromFile(char* img_path, int* _net, float threshold, float* result, int result_sz) {
	image im = load_image_color(img_path, 0, 0);
	int r=YoloDetect(im, _net, threshold, result, result_sz);
	free_image(im);
	return r;
}

DLL_MACRO int YoloDetectFromImage(float* data,int w,int h,int c, int* _net, float threshold, float* result, int result_sz) {
	image im;
	im.data=data;
	im.w=w;
	im.h=h;
	im.c=c;
	return YoloDetect(im, _net, threshold, result, result_sz);
}

#include<stdio.h>
#include<Windows.h>
typedef int*(*YoloLoadType)(char* cfg, char* weights);
typedef void(*YoloTrainType)(char* _base_dir, char* _datafile, char* _cfgfile);
typedef int(*YoloDetectFromFileType)(char* img_path, int* _net, float threshold, float* result, int result_sz);
int main() {
#ifdef _DEBUG
	HMODULE hMod = LoadLibraryA("YOLOv2SEd.dll");
#else
	HMODULE hMod = LoadLibraryA("YOLOv2SE.dll");
#endif
	if (hMod == NULL) {
		puts("dll load fail");
		exit(EXIT_FAILURE);
	}
	YoloLoadType YoloLoad = (YoloLoadType)GetProcAddress(hMod, "YoloLoad");
	YoloTrainType YoloTrain = (YoloTrainType)GetProcAddress(hMod, "YoloTrain");
	YoloDetectFromFileType YoloDetectFromFile = (YoloDetectFromFileType)GetProcAddress(hMod, "YoloDetectFromFile");

	int* network = YoloLoad("../../network/yolo.cfg", "../../network/yolo.weights");
	float result[2048] = { 0 };
	int sz = YoloDetectFromFile("../../test.jpg", network, 0.09F, result, 2048);
	for (int i = 0; i < sz; i++) {
		int kind = result[i * 6 + 0];
		int cval = result[i * 6 + 1];
		int left = result[i * 6 + 2];
		int top = result[i * 6 + 3];
		int width = result[i * 6 + 4];
		int height = result[i * 6 + 5];
		printf("%d\n", kind);
		printf("%d\n", cval);
		printf("%d\n", left);
		printf("%d\n", top);
		printf("%d\n", width);
		printf("%d\n", height);
	}
	FreeLibrary(hMod);
	return 0;
}
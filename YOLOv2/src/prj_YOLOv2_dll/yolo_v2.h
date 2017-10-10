
#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"curand.lib")
#pragma comment(lib,"cudart.lib")
#pragma comment(lib,"vfw32.lib")
#pragma comment( lib, "comctl32.lib" )
#pragma warning(disable:4099)
#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")

#include "network.h"

#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "option_list.h"
#include "stb_image.h"

#pragma once
#include<Windows.h>		//GetCurrentDirectoryA
#include<direct.h>		//_chdir
#include<opencv/cv.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<driver_types.h>
extern void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear);
int YoloDetect(image img, int* _net, float threshold, float* result, int result_sz);

__declspec(dllexport) void YoloTrain(char* _base_dir, char* _datafile, char* _cfgfile);
__declspec(dllexport) int* YoloLoad(char* cfgfile, char* weightsfile);
__declspec(dllexport) int YoloDetectFromFile(char* img_path, int* _net, float threshold, float* result, int result_sz);
__declspec(dllexport) int YoloDetectFromIplImage(IplImage* src, int* _net, float threshold, float* result, int result_sz);

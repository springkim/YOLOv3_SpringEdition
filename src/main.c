
#include"yolo_v2.h"
extern void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen);
char* GetCFGElement(char* cfg, char* name, char* buffer) {
	FILE* fp = fopen(cfg, "r");
	if (fp == NULL) {
		fprintf(stderr, "cfg error\n");
		return buffer;
	}
	char line[256 + 1];
	char temp[256 + 1];
	while (!feof(fp)) {
		fgets(line, 256, fp);
		if (strncmp(name, line, strlen(name)) == 0) {
			char* p = line + strlen(name);
			while (!isdigit(*p))p++;
			strcpy(buffer, p);
			break;
		}
	}
	fclose(fp);
	return buffer;
}
char* ParseCfg(char* cfg, char* buffer) {
	FILE* fp = fopen(cfg, "r");
	if (fp == NULL) {
		fprintf(stderr, "cfg error\n");
		return buffer;
	}
	char* base = basecfg(cfg);
	int width = 0;
	int random = 0;
	char line[256+1];
	while (!feof(fp)) {
		fgets(line, 256, fp);
		if (strncmp("width", line, 5) == 0) {
			sscanf(line, "width=%d", &width);
		}
		if (strncmp("random", line, 6) == 0) {
			sscanf(line, "random=%d", &random);
		}
	}
	fclose(fp);
	sprintf(buffer, "%s_%d", base, width);
	if (random == 1) {
		strcat(buffer, "_random");
	}
	return buffer;
}
/*
*	@YoloTrain
*	@param1 : 기준 디렉터리
*	@param2: data파일 이름(파일만 명시)
*	@param3 : cfg파일 이름(파일만 명시)
*/
void YoloTrain(char* _base_dir, char* _datafile, char* _cfgfile) {
	//모든 파일들은 base_dir을 기준으로 찾습니다.(상대 경로일경우 절대경로로 바꿔줍니다)
	char rpath[MAX_PATH];
	strcpy(rpath, _base_dir);
	char apath[MAX_PATH];
#if defined(_WIN32) || defined(_WIN64)
	_fullpath(apath, rpath, MAX_PATH);
#else
	realpath(rpath,apath);
#endif

	//현재 실행파일의 경로를 저장합니다.
	char curr_dir[MAX_PATH];
	chdir(curr_dir);
#if defined(_WIN32) || defined(_WIN64)
	GetCurrentDirectoryA(MAX_PATH, curr_dir);
	_chdir(apath);
#else
	getcwd(curr_dir,MAX_PATH);
	chdir(curr_dir);
#endif
	//콘솔응용프로그램이 아니거나, dll프로젝트일경우 콘솔창을 생성합니다.
#if defined(_WIN32) || defined(_WIN64)
	#if !defined(_CONSOLE) || defined(_WINDLL)
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
#endif
#endif
	int gpu_num = 0;
	cudaGetDeviceCount(&gpu_num);
	if (gpu_num == 0) {
		puts("No Gpus");
		return;
	}
	int* gpu_indexes = (int*)calloc(gpu_num, sizeof(int));
	int ngpu=0;
	char fastest_gpu[256]={0};
	for (int i = 0; i < gpu_num; i++){
		struct cudaDeviceProp cdp;
		cudaGetDeviceProperties(&cdp, i);
		if(i==0) {
			gpu_indexes[ngpu++] = i;
			printf("%d : %s\n", i, cdp.name);
			strcpy(fastest_gpu,cdp.name);
		}else{
			if(strncmp(fastest_gpu,cdp.name,strlen(cdp.name))==0){
				gpu_indexes[ngpu++] = i;
				printf("%d : %s\n", i, cdp.name);
			}else{
				printf("%d : %s(disable)\n", i, cdp.name);
			}
		}
	}
	char name[MAX_PATH];
	char* backup = NULL;
	char buffer[256] = { 0 };
	char* base = ParseCfg(_cfgfile, buffer);
	char _maxbatches[256] = { 0 };
	int maxbatches = atoi(GetCFGElement(_cfgfile, "max_batches", _maxbatches));

	for (int i = maxbatches; i >= 0;) {
		sprintf(name, "backup/%s_%d.weights", base,i);
#if defined(_WIN32) || defined(_WIN64)
		if (PathFileExistsA(name)==TRUE) {
#else
		if(access(name,F_OK)==0){
#endif
			printf("%s is lastest backup file",name);
			backup = name;
			break;
		}
		i -= 100;
	}
	puts("Train start!!");
	train_detector(
			_datafile
			, _cfgfile
			, backup
			, gpu_indexes
			, ngpu	/*number of gpu*/
			, 0,base);
#if defined(_WIN32) || defined(_WIN64)
	#if !defined(_CONSOLE) || defined(_WINDLL)
	FreeConsole();
#endif
	_chdir(curr_dir);
#else
	chdir(curr_dir);
#endif

}
int main(int argc, char** argv) {
	//dir , data , cfg
	//test_detector("voc.data", "darknet19.cfg", "darknet19.weights", "a.jpg", 0.5F, 0.5F, "b.jpg", 0);
	if(argc<4){
		puts("[arguments] working dir, data file, cfg file");
		return 1;
	}
	YoloTrain(argv[1], argv[2], argv[3]);
	return 0;
}

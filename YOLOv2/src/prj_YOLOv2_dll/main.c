#ifndef _WINDLL
#include"stdafx.h"
#include"yolo_v2.h"

int main() {
	YoloTrain("../", "obj.data", "obj.cfg");

	return 0;
}
#endif
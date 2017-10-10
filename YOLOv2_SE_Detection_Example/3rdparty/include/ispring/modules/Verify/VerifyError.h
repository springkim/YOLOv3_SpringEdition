/**
* @file		VerifyError.h
* @author		kimbomm (springnode@gmail.com)
* @date		2017. 05. 25...
* @version	1.0.0
*
*  @brief
*			에러 처리 라이브러리
*	@remark
*			Created by kimbom on 2017. 05. 25...
*			Copyright 2017 kimbom.All rights reserved.
*/
#if !defined(ISPRING_7E1_05_19_VERIFY_H_INCLUDED)
#define ISPRING_7E1_05_19_VERIFY_H_INCLUDED
#include<string>
#include<sstream>
#include<fstream>
#ifndef DOXYGEN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include<Windows.h>
#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")
#ifndef DOXYGEN
inline void __VerifyError(std::string msg, const char* __file__, const char* __function__, int __line__) {
	std::ostringstream oss;
	oss << "Verify Error : (" << msg << ") in " << __function__ << " , " << __file__ << " , line" << __line__ << std::endl;
	if (MessageBoxA(NULL, oss.str().c_str(), "SCVL Error", MB_RETRYCANCEL) == IDCANCEL) {
		exit(1);
	}
}
inline void __VerifyPointer(void* ptr, const char* __file__, const char* __function__, int __line__) {
	if (ptr == nullptr) {
		__VerifyError("Pointer is NULL", __file__, __function__, __line__);
	}
}
inline void __VerifyFilePath(std::string path, const char* __file__, const char* __function__, int __line__) {
	if (path.length() == 0) {
		__VerifyError("path is empty", __file__, __function__, __line__);
	}
	if (PathFileExistsA(path.c_str()) != TRUE) {
		__VerifyError("path file is incorrect", __file__, __function__, __line__);
	}
	std::fstream fchk(path, std::ios::in);
	bool can_not_open = fchk.is_open();
	fchk.close();
	if (can_not_open == false) {
		__VerifyError("can't open file", __file__, __function__, __line__);
	}
}
#endif
/**
*	@brief FATAL에러를 출력합니다. 출력후 계속 프로그램을 진행할지 종료할지 결정합니다.
*	@param MSG 화면에 출력할 에러 메세지 입니다.
*/
#define ISPRING_VERIFY(MSG)	__VerifyError(MSG,__FILE__,__FUNCTION__,__LINE__)

/**
*	@brief 포인터가 NULL 이면 FATAL에러를 출력합니다. 출력후 계속 프로그램을 진행할지 종료할지 결정합니다.
*/
#define ISPRING_VERIFY_POINTER(PTR)	__VerifyPointer(PTR,__FILE__,__FUNCTION__,__LINE__)

/**
*	@brief 파일이 존재하지 않으면 FATAL에러를 출력합니다. 출력후 계속 프로그램을 진행할지 종료할지 결정합니다.
*/
#define ISPRING_VERIFY_FILEPATH(PATH)	__VerifyFilePath(PATH,__FILE__,__FUNCTION__,__LINE__)


#endif
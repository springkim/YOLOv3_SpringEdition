/**
* @file		www.h
* @author		kimbomm (springnode@gmail.com)
* @date		2017. 10. 05...
* @version	1.0.0
*
*  @brief
*			HTML 처리 라이브러리
*	@remark
*			Created by kimbom on 2017. 10. 05...
*			Copyright 2017 kimbom.All rights reserved.
*/
#if !defined(ISPRING_7E1_A_4_HTML_HPP_INCLUDED)
#define ISPRING_7E1_A_4_HTML_HPP_INCLUDED
#include<iostream>
#include<vector>
#include<string>
#include<WinInet.h>
#include<Windows.h>
#include<urlmon.h>            //URLDownloadToFileA
#pragma comment(lib,"urlmon.lib")
#pragma comment(lib, "wininet.lib")
namespace ispring {
	/**
	*	@brief 이 정적 클래스는 웹을 다루는 함수를 포함합니다.
	*	@author kimbomm
	*	@date 2017-10-05
	*/
	class Web {
	public:
		/**
		*	@brief url의 html을 가져 옵니다.
		*	@param url URL
		*	@return html
		*/
		static std::string GetHtml(std::string url) {
			std::string html;
			HINTERNET hInternet = InternetOpenA("HTTP", INTERNET_OPEN_TYPE_PRECONFIG, NULL, NULL, 0);	//인터넷 관련 DLL을 초기화한다.
			if (hInternet) {
				HINTERNET hUrl = InternetOpenUrlA(hInternet, url.c_str(), NULL, 0, INTERNET_FLAG_RELOAD, 0);	//url에 걸린 파일을 연다.
				if (hUrl) {
					DWORD realSize = 0;
					DWORD possibleSize = 0;
					DWORD recv = 0;
					char* buffer = new char[2000000];
					char* tempBuffer = new char[2000000];
					memset(buffer, 0, 2000000 * sizeof(char));
					memset(tempBuffer, 0, 2000000 * sizeof(char));
					do {
						InternetQueryDataAvailable(hUrl, &possibleSize, 0, 0);
						InternetReadFile(hUrl, buffer, possibleSize, &realSize);
						if (possibleSize>0) {
							memcpy(tempBuffer + recv, buffer, possibleSize * sizeof(char));
						}
						recv += possibleSize;
					} while (realSize != 0);
					html.resize(realSize);
					html = tempBuffer;
					delete[] buffer;
					delete[] tempBuffer;
				}
			}
			return html;
		}
		/**
		*	@brief url의 파일을 다운로드 합니다.
		*	@param url URL
		*	@param file 저장할 파일 명
		*	@return 다운로드에 성공하면 true
		*/
		static bool Download(std::string url,std::string file) {
			HRESULT r = URLDownloadToFileA(nullptr, url.c_str(), file.c_str(), 0, 0);
			return r == S_OK;
		}
	};
}

#endif  //ISPRING_7E1_A_4_HTML_HPP_INCLUDED
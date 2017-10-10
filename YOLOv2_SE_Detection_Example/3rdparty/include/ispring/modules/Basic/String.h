/**
* @file		String.h
* @author		kimbomm (springnode@gmail.com)
* @date		2017. 10. 3...
* @version	1.0.0	
*
*  @brief
*			문자열 처리 라이브러리
*	@remark
*			Created by kimbom on 2017. 10. 3...
*			Copyright 2017 kimbom.All rights reserved.
*/
#if !defined(ISPRING_7E1_A_3_STRING_HPP_INCLUDED)
#define ISPRING_7E1_A_3_STRING_HPP_INCLUDED
#include<iostream>
#include<vector>
#include<string>
#include<sstream>
namespace ispring {
	/**
	*	@brief 이 정적 클래스는 문자열을 조작하는 함수를 포함합니다.
	*	@author kimbomm
	*	@date 2017-10-03
	*/
	class String {
	public:
		/**
		*	@brief 문자열을 토큰을 기준으로 분리합니다.
		*	@param str 입력 문자열
		*	@param token  토큰
		*	@return 잘려진 문자열들이 std::vector에 담겨서 반환됩니다.
		*	@warning 길이가 0인 문자열은 추가되지 않습니다.
		*	@details 예시 입력 :  ";*.cpp;*.h;*.c;*.jpg;;;;;" \n
				      예시 출력 :  {"*.cpp","*.h","*.c","*.jpg"}
		*/
		static std::vector<std::string> Tokenizer(std::string str, std::string token) {
			std::vector<std::string> ret;
			std::string::size_type offset = 0;
			while (offset < str.length()) {
				std::string word = str.substr(offset, str.find(token, offset) - offset);
				offset += word.length() + 1;
				if (word.length() > 0) {
					ret.push_back(word);
				}
			}
			return ret;
		}
		/**
		*	@brief 파일을 포함한 경로에서 확장자를 추출 합니다.
		*	@param path 파일 경로
		*	@return 확장자( . 은 포함되지 않음)
		*	@warning 확장자가 없으면 빈 문자열이 반환 됩니다.
		*/
		static std::string GetExtOfFile(std::string path) {
			std::string::size_type dot = path.find_last_of('.');
			if (dot == std::string::npos) {
				return "";
			}
			std::string ext = path.substr(dot+1, path.length() - dot-1);
			return ext;
		}
		/**
		*	@brief 파일의 이름을 경로에서 추출합니다.(확장자 포함)
		*	@param path 파일 경로
		*	@return 파일의 이름.
		*/
		static std::string GetNameOfFile(std::string path) {
			std::string::size_type slash = path.find_last_of("/\\");
			if (slash == std::string::npos) {
				slash = -1;
			}
			std::string name = path.substr(slash + 1, path.length() - slash - 1);
			return name;
		}
		/**
		*	@brief 파일을 포함한 경로에서 순수 파일의 이름을 반환 합니다(확장자 제외)
		*	@param path 파일 경로
		*	@return 순수 파일 이름(확장자 제외)
		*/
		static std::string GetPureNameOfFile(std::string path) {
			std::string::size_type slash = path.find_last_of("/\\");
			if (slash == std::string::npos) {
				slash = -1;
			}
			std::string::size_type dot = path.find_last_of(".");
			if (dot == std::string::npos) {
				dot = path.length();
			}
			std::string pure = path.substr(slash+1, dot - slash-1);
			return pure;
		}
		/**
		*	@brief 숫자를 0 패딩을 주어 반환 합니다.
		*	@param num 변환할 숫자
		*	@param pad 지정할 자릿수
		*	@return 패딩이 지정된 숫자에 대한 문자열
		*	@details 12를 4의 패딩을 주면 "0012" 라는 문자열이 반환됩니다.
		*/
		static std::string PadNum(unsigned int num,unsigned int pad) {
			std::ostringstream oss;
			oss.width(pad);
			oss.fill('0');
			oss << num;
			return oss.str();
		}
	};
}
#endif  //ISPRING_7E1_A_3_STRING_HPP_INCLUDED
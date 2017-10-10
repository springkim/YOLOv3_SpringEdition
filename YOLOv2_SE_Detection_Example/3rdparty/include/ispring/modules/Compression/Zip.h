/**
* @file		Zip.h
* @author		kimbomm (springnode@gmail.com)
* @date		2017. 09. 21...
* @version	1.0.0
*
*  @brief
*			Zip포맷 압축, 압축 해제 라이브러리
*	@remark
*			Created by kimbom on 2017. 9. 21...
*			Copyright 2017 kimbom.All rights reserved.
*/
#include"Winzip.h"
namespace ispring {
	/**
	*	@brief 이 정적 클래스는 Zip 포맷을 다룹니다.
	*	@author kimbomm
	*	@date 2017-09-21
	*/
	class Zip {
	public:
		/**
		*	@brief 폴더를 압축 합니다.
		*	@param folder 압축할 폴더의 경로
		*	@param _password 압축 파일에 설정할 비밀번호. (기본값은 비밀번호 없음)
		*	@param progress 압축의 진행 상황을 외부에서 알아올 수 있음.
		*	@return void
		*	@warning 이 압축 함수는 Bandizip의 기본 포맷을 따릅니다. 폴더를 압축하는 경우 폴더는 압축에 포함하지 않습니다.
		*	@remark
		*	@code{.cpp}
		*		puts("make zip_test folder...");
		*		_mkdir("zip_test");
		*		for (int i = 0; i < 100; i++) {
		*		std::ofstream fout("zip_test/" + std::to_string(i) + ".txt");
		*		for (int j = 0; j < 10000; j++) {
		*		fout << i << "\t" << j << std::endl;
		*		}
		*		fout.close();
		*		}
		*		puts("zip test...");
		*		ispring::Zip::Compress("zip_test");
		*		puts("unzip test...");
		*		ispring::File::DeleteDirectory("zip_test");
		*		ispring::Zip::Uncompress("zip_test.zip");
		*		ispring::File::DeleteDirectory("zip_test");
		*		DeleteFileA("zip_test.zip");
		*	@endcode
		*/
		static void Compress(std::string folder, std::string _password = "", std::atomic<int>* progress = nullptr) {
			if (PathFileExistsA(std::string(folder + ".zip").c_str()) == TRUE) {
				DeleteFileA(std::string(folder + ".zip").c_str());
			}
			std::vector<std::string> files = ispring::File::FileList(folder, "*.*", true);
#ifdef UNICODE
			std::wstring str(folder.begin(), folder.end());
			str += L".zip";
#else
			std::string str = folder + ".zip";
#endif
			const char* password = _password.length() > 0 ? _password.c_str() : nullptr;
			ispring_3rdparty::HZIP hz = ispring_3rdparty::CreateZip(str.c_str(), password);
			int m_progress = 0;
			for (size_t i = 0; i < files.size(); i++) {
#ifdef UNICODE
				std::wstring file(files[i].begin(), files[i].end());
				std::wstring dfile = file.substr(file.find_first_of(L"\\/") + 1, file.length());
#else
				std::string file(files[i].begin(), files[i].end());
				std::string dfile = file.substr(file.find_first_of("\\/") + 1, file.length());
#endif
				ispring_3rdparty::ZipAdd(hz, dfile.c_str(), file.c_str());
				if (progress != nullptr) {
					if ((float)i / files.size() * 100 != m_progress) {
						m_progress = static_cast<int>((float)i / files.size() * 100);
						*progress = m_progress;
					}
				}
			}
			if (progress != nullptr) {
				*progress = 100;
			}
			CloseZip(hz);
		}
		/**
		*	@brief 파일들을 압축 합니다.
		*	@param zip_name 압축 파일의 이름
		*	@param files 압축될 파일의 경로들
		*	@param _password 압축 파일에 설정할 비밀번호. (기본값은 비밀번호 없음)
		*	@param progress 압축의 진행 상황을 외부에서 알아올 수 있음.
		*	@return void
		*/
		static void Compress(std::string zip_name, std::vector<std::string> files, std::string _password = "", std::atomic<int>* progress = nullptr) {
			if (PathFileExistsA(zip_name.c_str()) == TRUE) {
				DeleteFileA(zip_name.c_str());
			}
#ifdef UNICODE
			std::wstring str(zip_name.begin(), zip_name.end());
			str += L".zip";
#else
			std::string str = zip_name + ".zip";
#endif
			const char* password = _password.length() > 0 ? _password.c_str() : nullptr;
			ispring_3rdparty::HZIP hz = ispring_3rdparty::CreateZip(str.c_str(), password);
			int m_progress = 0;
			for (size_t i = 0; i < files.size(); i++) {
#ifdef UNICODE
				std::wstring file(files[i].begin(), files[i].end());
#else
				std::string file(files[i].begin(), files[i].end());
#endif
				ispring_3rdparty::ZipAdd(hz, file.c_str(), file.c_str());
				if (progress != nullptr) {
					if ((float)i / files.size() * 100 != m_progress) {
						m_progress = static_cast<int>((float)i / files.size() * 100);
						*progress = m_progress;
					}
				}
			}
			if (progress != nullptr) {
				*progress = 100;
			}
			CloseZip(hz);
		}
		/**
		*	@brief 압축을 해제 합니다.
		*	@param file 해제할 압축 파일의 이름
		*	@param _password 압축해제에 필요한 비밀번호
		*	@param progress 압축해제의 진행 상황을 외부에서 알아올 수 있음.
		*	@return void
		*/
		static void Uncompress(std::string file, std::string _password = "", std::atomic<int>* progress = nullptr) {
			ISPRING_VERIFY_FILEPATH(file);
			std::string pure1 = file.substr(0, file.find_last_of('.')) + '/';
			if (PathFileExistsA(pure1.c_str()) == TRUE) {
				ispring::File::DirectoryErase(pure1.c_str());
			}
#ifdef UNICODE
			std::wstring str(file.begin(), file.end());
			std::wstring pure = str.substr(0, str.find_last_of(L'.')) + L'/';
			std::wstring add;
#else
			std::string str = file;
			std::string pure = str.substr(0, str.find_last_of('.')) + '/';
			std::string add;
#endif
			const char* password = _password.length() > 0 ? _password.c_str() : nullptr;
			ispring_3rdparty::HZIP hz = ispring_3rdparty::OpenZip(str.c_str(), password);
			ispring_3rdparty::ZIPENTRY ze;
			ispring_3rdparty::GetZipItem(hz, -1, &ze);
			int N = ze.index;
			int m_progress = 0;
			for (int i = 0; i < N; i++) {
				ispring_3rdparty::GetZipItem(hz, i, &ze);
				add = pure + ze.name;
				ispring_3rdparty::UnzipItem(hz, i, add.c_str());
				if (progress != nullptr) {
					if ((float)i / N * 100 != m_progress) {
						m_progress = static_cast<int>((float)i / N * 100);
						*progress = m_progress;
					}
				}
			}
			if (progress != nullptr) {
				*progress = 100;
			}
			CloseZip(hz);
		}
	};
}
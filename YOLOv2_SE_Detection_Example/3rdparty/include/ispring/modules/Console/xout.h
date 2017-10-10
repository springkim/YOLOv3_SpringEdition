/**
* @file		xout.h
* @author		kimbomm (springnode@gmail.com)
* @date		2017. 10. 3...
* @version	1.0.0
*
*  @brief
*			다중 콘솔 출력 라이브러리
*	@remark
*			Created by kimbom on 2017. 9. 20...
*			Copyright 2017 kimbom.All rights reserved.
*/
#if !defined(ISPRING_7E1_9_14_XOUT_HPP_INCLUDED)
#define ISPRING_7E1_9_14_XOUT_HPP_INCLUDED
#include"../defines.h"


#ifndef DOXYGEN
//https://stackoverflow.com/questions/10015897/cannot-have-typeofstdendl-as-template-parameter
template<class e, class t, class a> //string version
auto get_endl(const std::basic_string<e, t, a>&)
-> decltype(&std::endl<e, t>) {
	return std::endl<e, t>;
}
template<class e, class t> //stream version
auto get_endl(const std::basic_ostream<e, t>&)-> decltype(&std::endl<e, t>) {
	return std::endl<e, t>;
}
#endif

#ifdef _WIN32
#ifndef DOXYGEN
#include<iostream>
#include<string>
#include<Windows.h>
#include<sstream>
#include<fstream>
#include<map>
#include<ctime>
#include<direct.h>
#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib") //g++ [file] -lshlwapi

#if _MSC_VER==1910		///VS 2017
//https://i.imgur.com/TOFHpX4.png
#if _WIN64
const char* compiler = "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvarsall.bat amd64 ";
#else
const char* compiler = "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvarsall.bat x86 ";
#endif
#elif _MSC_VER==1900	///VS 2015
const char* compiler = "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/vcvarsall.bat";
#elif _MSC_VER==1800		///VS 2013
const char* compiler = "C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/vcvarsall.bat";
#elif _MSC_VER==1700		///VS2012
const char* compiler = "C:/Program Files (x86)/Microsoft Visual Studio 11.0/VC/vcvarsall.bat";
#elif _MSC_VER==1600		///VS2010
const char* compiler = "C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/vcvarsall.bat";
#elif defined(__GNUC__) ///MinGW
const char* compiler = "g++";
#endif
#endif //DOXYGEN
namespace ispring { 
#ifndef DOXYGEN
	class _xout;
	class _xout_color {
	public:
		unsigned char m_c;
		explicit _xout_color(unsigned char c) {
			m_c = c;
		}
	};
#endif
	/**
	*	@brief 이 스트림 클래스는 새로 생성한 클래스에 대한 출력 스트림 입니다.
	*	@details 자세한 내용은 _xout 클래스를 참조 하십시오
	*	@author kimbomm
	*	@date 2017-10-03
	*/
	class xstream {
	private:
		std::string m_key_mutex;
		HANDLE	m_mutex = INVALID_HANDLE_VALUE;
		std::string m_key_shmem;
		HANDLE	m_shmem = INVALID_HANDLE_VALUE;
		DWORD m_size;
		char* m_buffer;
		//[0]= 0:writable , 1:not-reading, 2:exit ('0'~'9','A'~'F') color change
		//[1]=not-use
		//[2]=not-use
	private:
		friend _xout;
		xstream(std::string key_mutex, std::string key_shmem, DWORD bufsize) {
			m_size = bufsize + 4;
			m_key_mutex = key_mutex;
			m_key_shmem = key_shmem;
			// https://msdn.microsoft.com/ko-kr/library/windows/desktop/ms682411(v=vs.85).aspx
			m_mutex = ::CreateMutexA(NULL, FALSE, m_key_mutex.c_str());
			if (m_mutex == NULL) {
				char* msg = NULL;
				FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER, 0, GetLastError(), 0, msg, 0, NULL);
				ISPRING_VERIFY(msg);
			}
			//https://msdn.microsoft.com/ko-kr/library/windows/desktop/aa366537(v=vs.85).aspx
			m_shmem = ::CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, m_size, m_key_shmem.c_str());
			if (m_shmem == NULL) {
				char* msg = NULL;
				FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER, 0, GetLastError(), 0, msg, 0, NULL);
				ISPRING_VERIFY(msg);
			}
			m_buffer = (char*)::MapViewOfFile(m_shmem, FILE_MAP_ALL_ACCESS, 0, 0, m_size);
		}
		~xstream() {
			bool write = false;
			while (write == false) {
				WaitForSingleObject(m_mutex, INFINITE);
				if (m_buffer[0] == 0) {
					m_buffer[0] = 2;
					write = true;
				}
				::ReleaseMutex(m_mutex);
			}
			::UnmapViewOfFile(m_buffer);
			::CloseHandle(m_shmem);
			::ReleaseMutex(m_mutex);
		}
	public:
		/**
		*	@brief 새 콘솔에 메세지를 출력합니다.
		*	@param msg 이 값은 char,int,float,double,std::string이 들어갈 수 있습니다.
		*	@return xstream&
		*/
		xstream& operator<<(char msg) {
			return this->operator<<(std::to_string(msg));
		}
		/**
		*	@brief 새 콘솔에 메세지를 출력합니다.
		*	@param msg 이 값은 char,int,float,double,std::string이 들어갈 수 있습니다.
		*	@return xstream&
		*/
		xstream& operator<<(int msg) {
			return this->operator<<(std::to_string(msg));
		}
		/**
		*	@brief 새 콘솔에 메세지를 출력합니다.
		*	@param msg 이 값은 char,int,float,double,std::string이 들어갈 수 있습니다.
		*	@return xstream&
		*/
		xstream& operator<<(float msg) {
			return this->operator<<(std::to_string(msg));
		}
		/**
		*	@brief 새 콘솔에 메세지를 출력합니다.
		*	@param msg 이 값은 char,int,float,double,std::string이 들어갈 수 있습니다.
		*	@return xstream&
		*/
		xstream& operator<<(double msg) {
			return this->operator<<(std::to_string(msg));
		}
		/**
		*	@brief 새 콘솔에 메세지를 출력합니다.
		*	@param msg 이 값은 char,int,float,double,std::string이 들어갈 수 있습니다.
		*	@return xstream&
		*/
		xstream& operator<<(std::string msg) {
			const char* begin = msg.c_str();
			bool write = false;
			while (write == false) {
				WaitForSingleObject(m_mutex, INFINITE);
				if (m_buffer[0] == 0) {
					m_buffer[0] = 1;
					write = true;
#ifdef _MSC_VER
					strcpy_s(m_buffer + 3, m_size, begin);
#elif defined(__GNUC__)
					strncpy(m_buffer + 3, begin, m_size);
#endif
				}
				::ReleaseMutex(m_mutex);
			}
			return *this;
		}
		/**
		*	@brief 새 콘솔에 색상을 설정합니다.
		*	@param c 색상입니다. 
		*	@return xstream&
		*/
		xstream& operator<<(_xout_color c) {
			bool write = false;
			while (write == false) {
				WaitForSingleObject(m_mutex, INFINITE);
				if (m_buffer[0] == 0) {
					m_buffer[0] = c.m_c;
					write = true;
					m_buffer[3] = 0;
				}
				::ReleaseMutex(m_mutex);
			}
			return *this;
		}
	};
	/**
	*	@brief 이 클래스는 선언 할 수 없으며, 미리 제공된 전역 객체 xout을 사용합니다.
	*	@author kimbomm
	*	@date 2017-10-03
	*/
	SELECT_ANY class _xout {
	private:
		std::map<std::string, xstream*> m_xstream;
		std::string GetKey() {
			//https://stackoverflow.com/questions/10654258/get-millisecond-part-of-time
			SYSTEMTIME stime;
			FILETIME ftime;
			FILETIME ftime_stamp;
			GetSystemTimeAsFileTime(&ftime_stamp);
			FileTimeToLocalFileTime(&ftime_stamp, &ftime);
			FileTimeToSystemTime(&ftime, &stime);
			char buf[256];
#ifdef _MSC_VER
			sprintf_s(buf, "%d%d%d%d%d%d%d", stime.wYear, stime.wMonth, stime.wDay, stime.wHour, stime.wMinute, stime.wSecond, stime.wMilliseconds);
#elif defined(__GNUC__)
			sprintf(buf, "%d%d%d%d%d%d%d", stime.wYear, stime.wMonth, stime.wDay, stime.wHour, stime.wMinute, stime.wSecond, stime.wMilliseconds);
#endif
			return buf;
		}
		std::string Compile(bool _override = false) {
			std::string code = "\
#include<iostream>\n\
#include<string>\n\
#include<Windows.h>\n\
char* bkcolor[] = {\n\
\"color 0F\",\n\
\"color 1F\",\n\
\"color 2F\",\n\
\"color 3F\",\n\
\"color 4F\",\n\
\"color 5F\",\n\
\"color 6F\",\n\
\"color 70\",\n\
\"color 8F\",\n\
\"color 9F\",\n\
\"color A0\",\n\
\"color B0\",\n\
\"color C0\",\n\
\"color D0\",\n\
\"color E0\",\n\
\"color F0\",\n\
	};\n\
int main(int argc,const char* argv[]) {\n\
	const char* mutex_key = argv[1];\n\
	const char* shmem_key = argv[2];\n\
	const int size = atoi(argv[3]);\n\
	const char* title = argv[4];\n\
	const int background=atoi(argv[5]);\n\
	SetConsoleTitleA(title);\n\
	system(bkcolor[background]);\n\
	HANDLE	shmem = INVALID_HANDLE_VALUE;\n\
	HANDLE	mutex = INVALID_HANDLE_VALUE;\n\
	mutex = OpenMutexA(MUTEX_ALL_ACCESS, FALSE, mutex_key);\n\
	shmem = OpenFileMappingA(FILE_MAP_READ|FILE_MAP_WRITE, FALSE, shmem_key);\n\
	char *buf = (char*)::MapViewOfFile(shmem, FILE_MAP_ALL_ACCESS, 0, 0, size);\n\
	while(1) {\n\
		WaitForSingleObject(mutex, INFINITE);\n\
		if (buf[0] == 2) {\n\
			::ReleaseMutex(mutex);\n\
			break;\n\
		}\n\
		if(buf[0]==1){\n\
			buf[size - 1] = 0;\n\
			printf(buf+3);\n\
			memset(buf, 0, 1);\n\
		}\n\
		if('0'<=buf[0] && buf[0]<='9'){\n\
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), (buf[0]-48)|(background*16));\n\
			memset(buf, 0, 1);\n\
		}\n\
		if('A'<=buf[0] && buf[0]<='F'){\n\
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), (buf[0]-65+10)|(background*16));\n\
			memset(buf, 0, 1);\n\
		}\n\
		::ReleaseMutex(mutex);\n\
	}\n\
	::UnmapViewOfFile(buf);\n\
	::CloseHandle(shmem);\n\
	::ReleaseMutex(mutex);\n\
	return 0;\n\
}\n\
";
			char c_temp_path[MAX_PATH + 1];
			GetTempPathA(MAX_PATH, c_temp_path);
			std::string temp_path = c_temp_path;
			std::string exe = temp_path + "xout.exe";
			if (_override == true || PathFileExistsA(exe.c_str()) == FALSE) {
				char curr_dir[MAX_PATH];
				GetCurrentDirectoryA(MAX_PATH, curr_dir);
				_chdir(temp_path.c_str());


				std::string cpp = "xout.cpp";
				std::string _log = "xlog.txt";
				std::fstream fout(cpp, std::ios::out);
				fout << code;
				fout.close();
				//https://stackoverflow.com/questions/876239/how-can-i-redirect-and-append-both-stdout-and-stderr-to-a-file-with-bash
#ifdef _MSC_VER
				std::string compile_arg = "\"" + std::string(compiler) + "\" & cl /O2 " + cpp + " >>" + _log + " 2>&1 -o xout.exe";
#elif defined(__GNUC__)
				std::string compile_arg = "\"" + std::string(compiler) + " -O2 " + cpp + " >>" + _log + " 2>&1 -o xout.exe";
#endif
				int succ = system(compile_arg.c_str());
				_chdir(curr_dir);
			}
			return exe;
		}
	public:
		~_xout() {
			for (auto&e : m_xstream) {
				delete e.second;
			}
		}
		/**
		*	@brief 새 콘솔을 생성합니다.
		*	@param window 생성할 콘솔의 이름
		*	@param bk 생성할 콘솔의 배경색
		*	@param bufsize IPC 버퍼.
		*/
		void Create(std::string window, _xout_color bk = _xout_color(0), DWORD bufsize = 1000) {
			if (m_xstream.find(window) != m_xstream.end()) {
				ISPRING_VERIFY(window + " is already exist!");
			}
			std::string key_mutex = this->GetKey() + "mutex";
			std::string key_shmem = this->GetKey() + "shmem";
			m_xstream.insert(std::make_pair(window, new xstream(key_mutex, key_shmem, bufsize)));
			std::string exe = this->Compile(false);
			int bki = 0;
			if ('0' <= bk.m_c && bk.m_c <= '9') {
				bki = bk.m_c - 48;
			} else if ('A' <= bk.m_c && bk.m_c <= 'F') {
				bki = bk.m_c - 65 + 10;
			}
			//https://stackoverflow.com/questions/154075/using-the-start-command-with-parameters-passed-to-the-started-program
			std::string exe_arg = "start \"\" \"" + exe + "\" " + key_mutex + " " + key_shmem + " " + std::to_string(bufsize) + " " + window + " " + std::to_string(bki);
			system(exe_arg.c_str());
		}
		/**
		*	@brief 지정된 콘솔에 대한 스트림을 얻어옵니다.
		*	@param window 생성한 콘솔의 이름
		*	@return xstream&
		*	@remark
		*	@code{.cpp}
		*	ispring::xout.Create("window1");
		*	ispring::xout.Create("window2");
		*	ispring::xout.Create("window3");
		*
		*	std::cout << ispring::xout.light_green << "hello, world" << std::endl;
		*	ispring::xout["window1"] << ispring::xout.light_red << "hello, world" << std::endl;
		*	ispring::xout["window2"] << ispring::xout.light_aqua << "hello, world" << std::endl;
		*	ispring::xout["window3"] << ispring::xout.light_yellow << "hello, world" << std::endl;
		*	std::cout << ispring::xout.white;
		*	@endcode
		*/
		xstream& operator[](std::string window) {
			auto it = m_xstream.find(window);
			if (it == m_xstream.end()) {
				ISPRING_VERIFY(window + " is not created");
			}
			return *it->second;
		}
	public:
		_xout_color black = _xout_color('0');
		_xout_color blue = _xout_color('1');
		_xout_color green = _xout_color('2');
		_xout_color aqua = _xout_color('3');
		_xout_color red = _xout_color('4');
		_xout_color purple = _xout_color('5');
		_xout_color yellow = _xout_color('6');
		_xout_color white = _xout_color('7');
		_xout_color gray = _xout_color('8');
		_xout_color light_blue = _xout_color('9');
		_xout_color light_green = _xout_color('A');
		_xout_color light_aqua = _xout_color('B');
		_xout_color light_red = _xout_color('C');
		_xout_color light_purple = _xout_color('D');
		_xout_color light_yellow = _xout_color('E');
		_xout_color light_white = _xout_color('F');
	}xout;

}
#ifndef DOXYGEN
inline ispring::xstream& operator<<(ispring::xstream& __xout, decltype(std::endl<char, std::char_traits<char>>) endl) {
	__xout << "\n";
	return __xout;
}
inline std::ostream& operator<<(std::ostream& __cout, ispring::_xout_color c) {
	//https://stackoverflow.com/questions/8578909/how-to-get-current-console-background-and-text-colors
	CONSOLE_SCREEN_BUFFER_INFO info;
	int bk = 0;
	if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) {
		bk = info.wAttributes / 16 * 16;
	}
	if ('0' <= c.m_c && c.m_c <= '9') {
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), (c.m_c - 48) | bk);
	} else if ('A' <= c.m_c && c.m_c <= 'F') {
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), (c.m_c - 65 + 10) | bk);
	}
	return __cout;
}
#endif //DOXYGEN

#endif //_WIN32
#ifdef __linux__
#include<iostream>
#include<sstream>
#include<map>
#include<cstring>
#include<fstream>
#include<cerrno>

#include<unistd.h>
#include<sys/shm.h>
#include<sys/time.h>
#include<pthread.h>
//http://xenostudy.tistory.com/422

//set terminal title
//https://stackoverflow.com/questions/2218159/c-console-window-title
namespace ispring {
	class _xout_color {
	public:
		unsigned char m_c;
		explicit _xout_color(unsigned char c) {
			m_c = c;
		}
	};
	class xstream {
	private:
		key_t m_key_shmem;
		int m_shmem;
		pthread_mutex_t* m_mutex;
		pthread_mutexattr_t* m_mutex_attr;
		size_t m_size;
		char* m_buffer;

	public:
		xstream(key_t key_shmem, size_t bufsize) {
			m_size = sizeof(pthread_mutex_t) + sizeof(pthread_mutexattr_t) + 4 + bufsize;
			m_key_shmem = key_shmem;
			m_shmem = shmget(m_key_shmem, m_size, IPC_CREAT | 0666);
			//http://man7.org/linux/man-pages/man2/shmget.2.html
			if (m_shmem < 0) {
				ISPRING_VERIFY(strerror(errno));
			}
			void* p = shmat(m_shmem, nullptr, 0);
			if (p == (void*)-1) {
				ISPRING_VERIFY(strerror(errno));
			}
			m_mutex = (pthread_mutex_t*)p;
			m_mutex_attr = (pthread_mutexattr_t*)((char*)p + sizeof(pthread_mutex_t));
			m_buffer = (char*)p + sizeof(pthread_mutex_t) + sizeof(pthread_mutexattr_t);
			if (pthread_mutexattr_init(m_mutex_attr) != 0) {
				ISPRING_VERIFY("pthread_mutexattr_init fail");
			}
			if (pthread_mutexattr_setpshared(m_mutex_attr, PTHREAD_PROCESS_SHARED) != 0) {
				ISPRING_VERIFY("pthread_mutexattr_setshared fail");
			}
			if (pthread_mutex_init(m_mutex, m_mutex_attr) != 0) {
				ISPRING_VERIFY("pthread_mutex_init fail");
			}
		}
		~xstream() {
			bool write = false;
			while (write == false) {
				if (pthread_mutex_lock(m_mutex) == 0) {
					if (m_buffer[0] == 0) {
						m_buffer[0] = 2;
						write = true;
						m_buffer[3] = 0;
					}
				}
				pthread_mutex_unlock(m_mutex);
			}
			shmdt((void*)m_mutex);
		}
		xstream& operator<<(char msg) {
			return this->operator<<(std::to_string(msg));
		}
		xstream& operator<<(int msg) {
			return this->operator<<(std::to_string(msg));
		}
		xstream& operator<<(float msg) {
			return this->operator<<(std::to_string(msg));
		}
		xstream& operator<<(double msg) {
			return this->operator<<(std::to_string(msg));
		}
		xstream& operator<<(std::string msg) {
			bool write = false;
			while (write == false) {
				if (pthread_mutex_lock(m_mutex) == 0) {
					if (m_buffer[0] == 0) {
						m_buffer[0] = 1;
						write = true;
						strcpy(m_buffer + 3, msg.c_str());
					}

				}
				pthread_mutex_unlock(m_mutex);
			}
			return *this;
		}
		xstream& operator<<(_xout_color c) {
			bool write = false;
			while (write == false) {
				if (pthread_mutex_lock(m_mutex) == 0) {
					if (m_buffer[0] == 0) {
						m_buffer[0] = c.m_c;
						write = true;
						m_buffer[3] = 0;
					}
				}
				pthread_mutex_unlock(m_mutex);
			}
			return *this;
		}
	};
	__attribute__((weak)) class _xout {
	private:
		std::map<std::string, xstream*> m_xstream;
		key_t GetKey() {
			struct timeval val;
			struct tm* ptm;
			gettimeofday(&val, NULL);
			ptm = localtime(&val.tv_sec);
			char buf[9];
			sprintf(buf, "%02d%02d%05ld", ptm->tm_min, ptm->tm_sec, val.tv_usec);
			return (key_t)atoi(buf);
		}
		std::string Compile(bool _override = false) {
			std::string code = "#include<cstdio>\n#include<iostream>\n#include<cstring>\n#include<pthread.h>\n#include<sys/shm.h>\nconst char* palette[]={\"\\x1b[30m\",\"\\x1b[34m\",\"\\x1b[32m\",\"\\x1b[36m\",\"\\x1b[31m\",\"\\x1b[35m\",\"\\x1b[33m\",\"\\x1b[37m\",\"\\x1b[90m\",\"\\x1b[94m\",\"\\x1b[92m\",\"\\x1b[96m\",\"\\x1b[91m\",\"\\x1b[95m\",\"\\x1b[93m\",\"\\x1b[97m\",};int main(int argc,const char* argv[]) {key_t key_shmem=(key_t)atoi(argv[1]);int size=atoi(argv[2]);const char* title=argv[3];printf(\"%c]0;%s%c\", '\\033', title, '\\007');size=sizeof(pthread_mutex_t) + sizeof(pthread_mutexattr_t) + 4 + size;pthread_mutex_t* mutex;pthread_mutexattr_t* mutex_attr;char* buffer;int shmem=shmget(key_shmem,size,IPC_CREAT|0666);if(shmem<0){return 1;}void* p=shmat(shmem,nullptr,0);if(p==(void*)-1){return 1;}mutex=(pthread_mutex_t*)p;mutex_attr=(pthread_mutexattr_t*)((char*)p+sizeof(pthread_mutex_t));buffer=(char*)p+sizeof(pthread_mutex_t)+sizeof(pthread_mutexattr_t);while(true){if(pthread_mutex_lock(mutex)==0){if(buffer[0]==2){pthread_mutex_unlock(mutex);break;}if(buffer[0]==1) {buffer[size - 1] = 0;printf(buffer + 3);memset(buffer, 0, 1);}if('0'<=buffer[0] && buffer[0]<='9'){printf(palette[buffer[0]-48]);buffer[0]=0;}if('A'<=buffer[0] && buffer[0]<='F'){printf(palette[buffer[0]-65+10]);buffer[0]=0;}}pthread_mutex_unlock(mutex);}return 0;}";
			std::string tmp = "/tmp/";
			std::string exe = tmp + "xout";
			if (access(exe.c_str(), F_OK) != 0 || _override == true) {
				std::ofstream fout(tmp + "xout.cpp");
				fout << code;
				fout.close();
				char curr_dir[1024];
				getcwd(curr_dir, 1023);
				chdir("/tmp/");
				std::string compile_arg = "g++ -std=c++11 xout.cpp -lpthread -o xout >>xout.log 2>&1";
				int succ = system(compile_arg.c_str());
				chdir(curr_dir);
			}

			return exe;
		}
	public:
		~_xout() {
			for (auto&e : m_xstream) {
				delete e.second;
			}
		}
		void Create(std::string window, _xout_color color = _xout_color(0), size_t bufsize = 1000) {
			//https://askubuntu.com/questions/484993/run-command-on-anothernew-terminal-window
			if (m_xstream.find(window) != m_xstream.end()) {
				ISPRING_VERIFY(window + " is already exist!");
			}
			key_t key_shmem = this->GetKey();
			m_xstream.insert(std::make_pair(window, new xstream(key_shmem, bufsize)));
			std::string exe = this->Compile(false);
			std::string exe_arg = std::string("gnome-terminal -e \"/tmp/xout ") + std::to_string(key_shmem) + " " + std::to_string(bufsize) + " " + window + "\"";
			system(exe_arg.c_str());

		}
		xstream& operator[](std::string window) {
			auto it = m_xstream.find(window);
			if (it == m_xstream.end()) {
				ISPRING_VERIFY(window + " is not created");
			}
			return *it->second;
		}
	public:
		_xout_color black = _xout_color('0');
		_xout_color blue = _xout_color('1');
		_xout_color green = _xout_color('2');
		_xout_color aqua = _xout_color('3');
		_xout_color red = _xout_color('4');
		_xout_color purple = _xout_color('5');
		_xout_color yellow = _xout_color('6');
		_xout_color white = _xout_color('7');
		_xout_color gray = _xout_color('8');
		_xout_color light_blue = _xout_color('9');
		_xout_color light_green = _xout_color('A');
		_xout_color light_aqua = _xout_color('B');
		_xout_color light_red = _xout_color('C');
		_xout_color light_purple = _xout_color('D');
		_xout_color light_yellow = _xout_color('E');
		_xout_color light_white = _xout_color('F');
	}xout;
}
inline ispring::xstream& operator<<(ispring::xstream& __xout, decltype(std::endl<char, std::char_traits<char>>) endl) {
	__xout << "\n";
	return __xout;
}
inline std::ostream& operator<<(std::ostream& __cout, ispring::_xout_color c) {
	const char* palette[] = {
		"\x1b[30m",
		"\x1b[34m",
		"\x1b[32m",
		"\x1b[36m",
		"\x1b[31m",
		"\x1b[35m",
		"\x1b[33m",
		"\x1b[37m",
		"\x1b[90m",
		"\x1b[94m",
		"\x1b[92m",
		"\x1b[96m",
		"\x1b[91m",
		"\x1b[95m",
		"\x1b[93m",
		"\x1b[97m",
	};
	if ('0' <= c.m_c && c.m_c <= '9') {
		printf("%s", palette[c.m_c - 48]);
	} else if ('A' <= c.m_c && c.m_c <= 'F') {
		printf("%s", palette[c.m_c - 65 + 10]);
	}
	return __cout;
}
#endif	//__linux__

#endif  //ISPRING_7E1_9_14_XOUT_HPP_INCLUDED
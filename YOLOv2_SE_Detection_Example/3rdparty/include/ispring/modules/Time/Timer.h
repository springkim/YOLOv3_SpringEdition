/**
* @file		Timer.h
* @author		kimbomm (springnode@gmail.com)
* @date		2017. 05. 23...
* @version	1.0.0
*
*  @brief
*			시간 측정 라이브러리
*	@remark
*			Created by kimbom on 2017. 05. 23...
*			Copyright 2017 kimbom.All rights reserved.
*/
#if !defined(ISPRING_7E1_05_18_TIME_H_INCLUDED)
#define ISPRING_7E1_05_18_TIME_H_INCLUDED
#include"../defines.h"
#include<iostream>
#include<vector>
#include<string>
#include<deque>
#include<algorithm>
#include<utility>
#include<ctime>
#include<cstring>
#include<cstdio>
#include<chrono>
#include<typeinfo>
#include<list>
#include<cmath>
#pragma warning(disable:4290)
#include"../Verify/VerifyError.h"


namespace ispring {
#ifndef DOXYGEN
	using _TimerType = std::chrono::system_clock::time_point;
	using _TimerCount = long long;
#endif
	/**
	*	@brief 이 정적 클래스는 시간을 측정하는데 사용됩니다.
	*	@author kimbomm
	*	@date 2017-05-23
	*/
	class Timer {
		class TimerElement {
		public:
			TimerElement() {
				curr = 0.0;
				avg = 0.0;
				accu = 0.0;
			}
			double curr;
			double avg;
			double accu;
		};
#ifndef DOXYGEN
		class GenericType {
			friend class Timer;
		private:
			int idx = -1;
			char _1;
			int _2;
			double _3;
			std::string _4;

			template<typename T>
			void SetObject(T param) {
				if (typeid(this->_1).hash_code() == typeid(param).hash_code()) {
					this->idx = 1;
					memcpy(&this->_1, &param, sizeof(char));
				} else if (typeid(this->_2).hash_code() == typeid(param).hash_code()) {
					this->idx = 2;
					memcpy(&this->_2, &param, sizeof(int));
				} else if (typeid(this->_3).hash_code() == typeid(param).hash_code()) {
					this->idx = 3;
					memcpy(&this->_3, &param, sizeof(double));
				} else if (typeid(char*).hash_code() == typeid(param).hash_code()
						   || typeid(const char*).hash_code() == typeid(param).hash_code()
						   || typeid(std::string).hash_code() == typeid(param).hash_code()
						   ) {
					this->idx = 4;
					char* tmp;
					memcpy(&tmp, &param, sizeof(char*));
					this->_4 = tmp;
				}
			}
		public:
			bool operator==(GenericType& other)const {
				if (this->idx != other.idx) {
					return false;
				} else {
					bool ret = false;
					switch (this->idx) {
						case 1:ret = this->_1 == other._1; break;
						case 2:ret = this->_2 == other._2; break;
						case 3:ret = fabs(this->_3 - other._3) < 0.0001; break;
						case 4:ret = this->_4 == other._4; break;
						default:break;
					}
					return ret;
				}
			}
		};
#endif
	private:
		Timer() = delete;
		~Timer() = delete;
	private:
		static std::vector<GenericType> mapped;
		static std::vector<std::pair<double, _TimerCount>> accumulated;
		static std::vector<_TimerType> temp;
		static std::deque<GenericType> stk;
		static std::list<std::pair<GenericType, TimerElement>> watch_map;
		template<typename CONTAINER>
		static intptr_t GetPosInContainer(CONTAINER container, const GenericType& pts) {
			intptr_t idx = -1;
			while ((idx++) + 1 < (intptr_t)container.size()) {
				if (pts == container[idx] == true) {
					break;
				}
			}
			if (idx != -1 && idx < (intptr_t)container.size()) {
				return idx;
			} else {
				return -1;
			}
		}
	public:
		/**
		*	@brief 시간 측정을 시작합니다.
		*	@param param 시간 측정 구분자(이 값은 char,int,float,double,string)을 지원합니다.
		*	@warning 같은 구분자를 Tock이 불려지기전에 사용할 수 없습니다.
		*/
		template<typename T>
		static void Tick(T param) {
			using GENERIC_TYPE = typename std::enable_if<
				std::is_same<char, T>::value
				|| std::is_same<int, T>::value
				|| std::is_same<float, T>::value
				|| std::is_same<double, T>::value
				|| std::is_same<char*, T>::value
				|| std::is_same<const char*, T>::value
				|| std::is_same<std::string, T>::value
				, T>::type;
			GenericType pts;
			pts.SetObject(param);

			intptr_t stkidx = Timer::GetPosInContainer(Timer::stk, pts);
			if (stkidx != -1) {	//exist
				ISPRING_VERIFY("Startclock and endclock is not matched");
			}
			stk.push_back(pts);

			intptr_t idx = Timer::GetPosInContainer(Timer::mapped, pts);
			if (idx != -1) {
				Timer::temp[idx] = std::chrono::system_clock::now();
			} else {				
				//new parameter
				Timer::temp.push_back(std::chrono::system_clock::now());
				Timer::mapped.push_back(pts);
				accumulated.push_back(std::make_pair(0.0, 0));
			}
		}
		/**
		*	@brief 시간 측정을 죵료합니다. 
		*	@param param 시간 측정 구분자(이 값은 char,int,float,double,string)을 지원합니다.
		*	@return TimerElement구조체를 반환합니다. 이 구조체는 curr,avg,accu 를 가지고 있으며 현재 구간 시간, 평균 구간 시간, 누적 구간 시간 입니다.
		*	@warning 이 함수는 반드시 Tick이 불려진다음 호출되야 합니다. 
		*	@remark
		*	@code{.cpp}
		*	for (int i = 0; i < 5; i++) {
		*		ispring::Timer::Tick(1234);
		*		Sleep(1000);
		*		ispring::Timer::Tock(1234);
		*	}
		*	std::cout << ispring::Timer::Watch(1234).avg << std::endl;
		*	@endcode
		*/
		template<typename T>
		static Timer::TimerElement Tock(T param) {
			using GENERIC_TYPE = typename std::enable_if<
				std::is_same<char, T>::value
				|| std::is_same<int, T>::value
				|| std::is_same<float, T>::value
				|| std::is_same<double, T>::value
				|| std::is_same<char*, T>::value
				|| std::is_same<const char*, T>::value
				|| std::is_same<std::string, T>::value
				, T>::type;
			GenericType pts;
			pts.SetObject(param);
			intptr_t stkidx = Timer::GetPosInContainer(Timer::stk, pts);
			if (stkidx == -1) {	//exist
				ISPRING_VERIFY("There is no Tick of this object");
			}
			intptr_t depth = stkidx;
			Timer::stk.erase(Timer::stk.begin() + stkidx);
			intptr_t idx = Timer::GetPosInContainer(Timer::mapped, pts);
			Timer::TimerElement tm;

			if (idx != -1) {
				tm.curr = static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - Timer::temp[idx]).count();
				accumulated[idx].first += tm.curr;
				accumulated[idx].second++;
				tm.accu = accumulated[idx].first;
				tm.avg = tm.accu / accumulated[idx].second;
			} else {
				ISPRING_VERIFY("No matched begin entry point");
			}
			
			auto it = std::find_if(watch_map.begin(), watch_map.end(), [&pts](std::pair<GenericType, TimerElement>& e)->bool { return e.first == pts; });
			if (it == watch_map.end()) {
				watch_map.push_front(std::make_pair(pts, tm));
			} else {
				it->second = tm;
			}
			return tm;
		}
		/**
		*	@brief 측정된 시간을 반환합니다.
		*	@param param 시간 측정 구분자(이 값은 char,int,float,double,string)을 지원합니다.
		*	@return TimerElement구조체를 반환합니다. 이 구조체는 curr,avg,accu 를 가지고 있으며 현재 구간 시간, 평균 구간 시간, 누적 구간 시간 입니다.
		*	@warning 이 함수는 반드시 Tick과 Tock이 끝난후 불러야 합니다.
		*	@remark
		*			고전 방식의 시간 측정은 시간 측정을 위해, 변수를 항상 가지고 있어야 했습니다.\n
		*			이 방식은 내부적으로 변수를 저장하며, 최신 기능의 std::chrono를 사용하며 코드가 깔끔해지고 정확합니다.
		*/
		template<typename T>
		static Timer::TimerElement Watch(T param) {
			using GENERIC_TYPE = typename std::enable_if<
				std::is_same<char, T>::value
				|| std::is_same<int, T>::value
				|| std::is_same<float, T>::value
				|| std::is_same<double, T>::value
				|| std::is_same<char*, T>::value
				|| std::is_same<const char*, T>::value
				|| std::is_same<std::string, T>::value
				, T>::type;
			GenericType pts;
			pts.SetObject(param);
			auto it = std::find_if(watch_map.begin(), watch_map.end(), [&pts](std::pair<GenericType, TimerElement>& e)->bool { return e.first == pts; });
			if (it == watch_map.end()) {
				ISPRING_VERIFY("No matched clock entry point");
			}
			return it->second;
		}
	};
	SELECT_ANY std::vector<Timer::GenericType> Timer::mapped;
	SELECT_ANY std::vector<_TimerType> Timer::temp;
	SELECT_ANY std::deque<Timer::GenericType> Timer::stk;
	SELECT_ANY std::vector<std::pair<double, _TimerCount>> Timer::accumulated;
	SELECT_ANY std::list<std::pair<Timer::GenericType, Timer::TimerElement>> Timer::watch_map;
}
#endif
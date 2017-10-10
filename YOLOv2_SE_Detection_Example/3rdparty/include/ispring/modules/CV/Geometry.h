/**
* @file		Geometry.h
* @author		kimbomm (springnode@gmail.com)
* @date		2017. 9. 13...
* @version	1.0.0
*
*  @brief
*			간단한 기하학 라이브러리
*	@remark
*			Created by kimbom on 2017. 9. 13...
*			Copyright 2017 kimbom.All rights reserved.
*/
#if !defined(ISPRING_7E1_9_D_GEOMETRY_HPP_INCLUDED)
#define ISPRING_7E1_9_D_GEOMETRY_HPP_INCLUDED
#include"../Verify/VerifyError.h"
#include<iostream>
#include<vector>
#include<string>
#include<numeric>
#include<opencv2/opencv.hpp>
namespace ispring {
	/**
	*	@brief 이 정적 클래스는 좌표 계산을 도와 줍니다.
	*	@author kimbomm
	*	@date 2017-09-13
	*/
	class CVGeometry {
	public:
		/**
		*	@brief 두점 사이의 각도를 계산 합니다.
		*	@param x1 첫번째 x좌표
		*	@param y1 첫번째 y좌표
		*	@param x2 두번째 x좌표
		*	@param y2 두번째 y좌표
		*	@return 각도(0~179)
		*	@warning 결과는 0~179 사이만 출력 합니다.
		*/
		static int GetDegree(int x1, int y1, int x2, int y2) {
			int d = static_cast<int>(floor((atan2(y1 - y2, x2 - x1) * 180 / acos(-1)) + 0.5));
			return (d + 180) % 180;//각도 양수화
		}
		/**
		*	@brief 두점 사이의 각도를 계산 합니다.
		*	@param from 첫번째 좌표
		*	@param to 두번째 좌표
		*	@return 각도(0~179)
		*	@warning 결과는 0~179 사이만 출력 합니다.
		*/
		static int GetDegree(cv::Point from, cv::Point to) {
			return GetDegree(from.x, from.y, to.x, to.y);
		}
		/**
		*	@brief 두 각도의 차이를 계산합니다.
		*	@param d1 첫번째 각도
		*	@param d2 두번째 각도
		*	@return 각도의 최소 차이
		*	@remark
		*		<img src="https://i.imgur.com/ec4bEjV.png" width="640">
		*/
		static int GetDegreeDistance(int d1, int d2) {
			if (d1 < 0 || d1>179) {
				ISPRING_VERIFY("degree(d1) is less than 0 or greater than 179");
			}
			if (d2 < 0 || d2>179) {
				ISPRING_VERIFY("degree(d2) is less than 0 or greater than 179");
			}
			int dist1=std::max(d1, d2) - std::min(d1, d2);	//first dist
			int dist2 = 180 - std::max(d1, d2) + std::min(d1, d2);	//second dist
			return std::min(dist1, dist2);
		}
		/**
		*	@brief 점들로 최소자승법을 이용해 직선을 추출 합니다.
		*	@param chain 점들의 집합
		*	@param threshold1 이 값보다 RMSE가 큰 군집 까지만 직선으로 간주합니다.
		*	@param threshold2 픽셀들이 직선이 되기위한 최소 점
		*	@return 검출된 선들의 집합
		*/
		std::vector<std::pair<cv::Point, cv::Point>> FindLineByLeastSquareMethod(std::vector<cv::Point> chain, float threshold1 = 1.0F, int threshold2 = 5) {
			std::vector<std::pair<cv::Point, cv::Point>> ret;
			bool linked = false;
			std::vector<cv::Point> line;
			for (size_t i = 0; i < chain.size(); i++) {
				if (linked == false) {
					linked = true;
					line.clear();
					line.push_back(chain[i]);
				} else {
					line.push_back(chain[i]);
					cv::Point2f mean = std::accumulate(line.begin(), line.end(), cv::Point2f(0.0F, 0.0F), [](cv::Point2f a, cv::Point e) {return cv::Point2f(a.x + e.x, a.y + e.y); });
					mean.x/=(int)line.size();
					mean.y /= (int)line.size();
					float denominator = std::accumulate(line.begin(), line.end(), float(0.0F), [&mean](float a, cv::Point e)->float {return pow(e.x - mean.x, 2.0F); });
					float numerator = std::accumulate(line.begin(), line.end(), float(0.0F), [&mean](float a, cv::Point e)->float {return (e.x - mean.x)*(e.y - mean.y); });
					float M = denominator != 0 ? (float)numerator / denominator : std::numeric_limits<float>::infinity();
					float B = mean.y - M*mean.x;
					//y=Mx+B;
					float rmse = std::sqrt(std::accumulate(line.begin(), line.end(), float(), [&M, &B](float a, cv::Point e)->float {
						if (M == std::numeric_limits<float>::infinity()) {	//기울기가 무한대이면 x에 대해 올바른 y가 나왔다고 가정함.
							return a;
						}
						return a + std::pow(std::fabs(e.y - (e.x*M + B)), 2);
					}) / line.size());

					if (rmse >= threshold1 || i == chain.size() - 1) {
						linked = false;
						line.pop_back();
						i--;
						cv::Point begin_pt = line.front();
						cv::Point end_pt = line.back();
						if (static_cast<int>(line.size()) >= threshold2) {
							int degree = GetDegree(begin_pt, end_pt);
							if (degree < 70 || degree>110) {
								ret.push_back(std::make_pair(begin_pt, end_pt));
							}
						}
					}
				}
			}
			return ret;
		}
	private:
		static bool InOrOut(cv::Point pt, cv::Point line1, cv::Point line2) {
			return ((pt.x - line1.x)*(pt.y - line2.y) - (pt.x - line2.x)*(pt.y - line1.y)) < 0.0F;
		}
	public:
		/**
		*	@brief 점이 삼각형안에 있는지 판별합니다.
		*	@param pt 검사할 점
		*	@param t1		삼각형의 첫번째 점
		*	@param t2	삼각형의 두번째 점
		*	@param t3 	삼각형의 세번째 점
		*	@return 점이 삼각형에 포함되면 true.
		*/
		bool PtInTriangle(cv::Point pt, cv::Point t1, cv::Point t2, cv::Point t3) {
			bool b1, b2, b3;
			b1 = InOrOut(pt, t1, t2);
			b2 = InOrOut(pt, t2, t3);
			b3 = InOrOut(pt, t3, t1);
			return (b1 == b2) && (b2 == b3);
		}
		/**
		*	@brief 점이 사각형안에 있는지 판별합니다.
		*	@param pt 검사할 점
		*	@param t1		사각형의 첫번째 점
		*	@param t2	사각형의 두번째 점
		*	@param t3 	사각형의 세번째 점
		*	@param t4 	사각형의 네번째 점
		*	@return 점이 사각형에 포함되면 true.
		*/
		bool PtInRectangle(cv::Point pt, cv::Point t1, cv::Point t2, cv::Point t3, cv::Point t4) {
			bool b1, b2;
			b1 = PtInTriangle(pt, t1, t2, t3);
			b2 = PtInTriangle(pt, t1, t4, t3);
			return b1 | b2;
		}
	};
}

#endif  //ISPRING_7E1_9_D_GEOMETRY_HPP_INCLUDED
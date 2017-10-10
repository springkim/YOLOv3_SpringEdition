/**
* @file		Eval.h
* @author		kimbomm (springnode@gmail.com)
* @date		2017. 10. 05...
* @version	1.0.0
*
*  @brief
*			간단한 기하학 라이브러리
*	@remark
*			Created by kimbom on 2017. 10. 05...
*			Copyright 2017 kimbom.All rights reserved.
*/
#if !defined(ISPRING_7E1_A_5_EVAL_HPP_INCLUDED)
#define ISPRING_7E1_A_5_EVAL_HPP_INCLUDED
#include<iostream>
#include<vector>
#include<string>
#include<map>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include"Image.h"
#ifndef  SPRING_EDITION_BOX
#define SPRING_EDITION_BOX
/**
*	@brief 이 클래스는 cv::Rect를 확장한 것으로 클래스값과 스코어값이 추가되었습니다.
*	@author kimbomm
*	@date 2017-10-05
*
*	@see	https://github.com/springkim/FasterRCNN_SpringEdition
*	@see	https://github.com/springkim/YOLOv2_SpringEdition
*/
class BoxSE : public cv::Rect {
public:
	int m_class = -1;
	float m_score = 0.0F;
	std::string m_class_name;
	BoxSE() {
		m_class_name = "Unknown";
	}
	BoxSE(int c, float s, int _x, int _y, int _w, int _h, std::string name = "")
		:m_class(c), m_score(s) {
		this->x = _x;
		this->y = _y;
		this->width = _w;
		this->height = _h;
		char* lb[5] = { "th","st","nd","rd","th" };
		if (name.length() == 0) {
			m_class_name = std::to_string(m_class) + lb[m_class < 4 ? m_class : 4] + " class";
		}
	}
};
#endif
namespace ispring {
	/**
	*	@brief 이 정적 클래스는 평가계산을 도와 줍니다.
	*	@author kimbomm
	*	@date 2017-10-05
	*/
	class CVEval {
	public:
		/**
		*	@brief Intersection Of Union을 계산합니다.
		*	@param a 첫번째 사각형
		*	@param b 두번째 사각형
		*	@return IOU값 (0~1)
		*	@remark
		*		<img src="https://i.imgur.com/QPFcJDk.png" width="640">
		*/
		static float IOU(cv::Rect a, cv::Rect b) {
			int i = (a&b).area();
			int u = a.area() + b.area() - i;
			return (float)i / u;
		}
		/**
		*	@brief Recall과 Precision을 단일 클래스에 대해 계산 합니다. 
		*	@param ground_truth 실제 정답들
		*	@param predict 추측 값들
		*	@param threshold1 이 값보다 큰 IOU를 가지는 박스를 정답으로 간주합니다.
		*	@return pair.first는 Recall 이고 pair.second는 precision 입니다.
		*/
		static std::pair<float,float> GetRecallPrecisionSingleClass(std::vector<BoxSE> ground_truth,std::vector<BoxSE> predict,float threshold1=0.5F) {
			std::pair<float, float> ret = { 0.0F,0.0F };
			if (predict.size() == 0) {
				return ret;
			}
			std::vector<int> correct;
			correct.assign(predict.size(), -1);
			for (size_t i = 0; i < predict.size(); ++i) {
				for (size_t j = 0; j < ground_truth.size(); ++j) {
					if (IOU(predict[i], ground_truth[j]) > threshold1) {
						if (correct[i] == -1 || ground_truth[correct[i]].m_score < ground_truth[j].m_score) {
							correct[i] = static_cast<int>(j);
						}
					}
				}
			}
			std::vector<bool> bloom;
			bloom.assign(ground_truth.size(), false);
			for (auto&e : correct) {
				if (e != -1) {
					bloom[e] = true;
				}
			}
			auto TP = std::count(bloom.begin(), bloom.end(), true);
			auto FN = std::count(bloom.begin(), bloom.end(), false);
			auto FP = std::count(correct.begin(), correct.end(), -1);
			ret.first = (float)TP / (TP + FN);
			ret.second = (float)TP / (TP + FP);
			return ret;
		}
		/**
		*	@brief Recall과 Precision을 다중 클래스에 대해 계산 합니다.
		*	@param ground_truth 실제 정답들
		*	@param predict 추측 값들
		*	@param threshold1 이 값보다 큰 IOU를 가지는 박스를 정답으로 간주합니다.
		*	@return std::map의 KEY는 클래스의 인덱스 이고 VALUE는 pair<float,float> 로써 first는 Recall, Second는 Precision 입니다.
		*/
		static std::map<int, std::pair<float, float>> GetRecallPrecisionMultiClass(std::vector<BoxSE> ground_truth, std::vector<BoxSE> predict, float threshold1) {
			std::map<int, std::pair<float, float>> ret;
			int n = 0;		//number of class
			for (auto&e : ground_truth) {
				n = std::max(e.m_class, n);
			}
			for (int i = 0; i <= n; i++) {
				std::vector<BoxSE> gt;
				std::vector<BoxSE> pt;
				for (auto&e : ground_truth) {
					if (e.m_class == i) {
						gt.push_back(e);
					}
				}
				for (auto&e : predict) {
					if (e.m_class == i) {
						pt.push_back(e);
					}
				}
				if (gt.size() == 0 && pt.size() == 0) {
					continue;
				}
				//==
				std::pair<float, float> rp = GetRecallPrecisionSingleClass(gt, pt, threshold1);
				ret.insert(std::make_pair(i, rp));
			}
			return ret;
		}
		/**
		*	@brief BoxSE를 이미지에 라벨과 함께 그립니다.
		*	@param img 박스가 그려질 이미지
		*	@param box 박스
		*	@param c 색상. 지정하지 않으면 GetRGB 함수를 사용합니다.
		*	@param additional 라벨에 추가로 적을 문자열.
		*	@remark
		*		<img src="https://i.imgur.com/Vs9uqvM.jpg" width="640">
		*/
		static void DrawBoxSE(cv::Mat img, BoxSE box,cv::Scalar c=cv::Scalar(-1,-1,-1), std::string additional="") {
			std::string label = box.m_class_name;
			if (box.m_score >=0) {
				label += "(" + std::to_string(box.m_score).substr(0, 4) + ")";
			} else {
				label += "(GroundTruth)";
			}
			if (additional.length() > 0) {
				label += "{" + additional + "}";
			}
			if (c[0] == -1 || c[1] == -1 || c[2] == -1) {
				c = ispring::CV::GetRGB(box.m_class);
			}
			int font_face = cv::FONT_HERSHEY_SIMPLEX;
			int wh = std::max(img.cols, img.rows);
			double scale = std::max((float)wh / 3000, 0.5F);
			int thickness = static_cast<int>((float)wh / 2000 + 1);
			int baseline = 0;
			
			cv::Size text = cv::getTextSize(label, font_face, scale, thickness, &baseline);
			int padding = static_cast<int>(text.height*0.2);
			cv::Point or (box.x, box.y);
			cv::Scalar text_color = cv::Scalar(255, 255, 255);
			if ((c[0] + c[1] + c[2]) / 3 > 150) {
				text_color = cv::Scalar(0, 0, 0);
			}
			
			int t = std::max((wh / 1600) + 1, 2);
			if (box.m_score >= 0) {
				cv::rectangle(img, box, c, t);
			} else {
				cv::rectangle(img, box, text_color, t*2+1);
				cv::rectangle(img, box, c, t);
				
			}
			cv::rectangle(img, or +cv::Point(0, -text.height - padding * 2), or +cv::Point(text.width, 0), c, CV_FILLED);
			cv::putText(img, label, or +cv::Point(0, -padding), font_face, scale, text_color, thickness);
		}
	};
}
#endif  //ISPRING_7E1_A_5_EVAL_HPP_INCLUDED
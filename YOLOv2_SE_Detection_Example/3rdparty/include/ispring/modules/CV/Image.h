/**
* @file		Image.h
* @author		kimbomm (springnode@gmail.com)
* @date		2017. 05. 23...
* @version	1.0.0
*
*  @brief
*			OpenCV를 이용한 이미지 처리 라이브러리
*	@remark
*			Created by kimbom on 2017. 05. 23...
*			Copyright 2017 kimbom.All rights reserved.
*/
#if !defined(ISPRING_7E1_05_17_IMAGE_H_INCLUDED)
#define ISPRING_7E1_05_17_IMAGE_H_INCLUDED
#include"../defines.h"
#include"rgb.h"
#include<opencv2/opencv.hpp>
#ifndef DOXYGEN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include<Windows.h>
#include"../Verify/VerifyError.h"
#include"../System/OS.h"
namespace ispring {
	/**
	*	@brief 이 정적 클래스는 이미지 처리 함수를 포함합니다.
	*	@author kimbomm
	*	@date 2017-05-23
	*/
	class CV{
	public:
		/**
		*	@brief 이진 이미지를 세선화 합니다.
		*	@param image 입력 이미지
		*	@warning 입력 이미지는 반드시 이진 이미지여야 합니다.
		*	@return 세선화된 이미지
		*	@remark
		*		<img src="https://i.imgur.com/qCUoy5U.jpg" width="640">
		*/
		static cv::Mat Thinning(cv::Mat image) {
			//https://github.com/bsdnoobz/zhang-suen-thinning
			if (image.channels() != 1 || image.depth() != 0) {
				ISPRING_VERIFY("This is not binary image");
			}
			if (image.rows <= 3 || image.cols <= 3) {
				ISPRING_VERIFY("Image size is too small");
			}
			cv::Mat ret = image.clone();
			ret /= 255;
			cv::Mat prev_img = cv::Mat::zeros(ret.size(), CV_8UC1);
			cv::Mat marker = cv::Mat::zeros(ret.size(), CV_8UC1);
			cv::Mat diff;
			do {
				for (int iter = 0; iter < 2; iter++) {
					marker.setTo(0);
					int nRows = ret.rows;
					int nCols = ret.cols;
					if (ret.isContinuous() == true) {
						nCols *= nRows;
						nRows = 1;
					}
					int x, y;
					uchar *pAbove;
					uchar *pCurr;
					uchar *pBelow;
					uchar *nw, *no, *ne;    // north (pAbove)
					uchar *we, *me, *ea;
					uchar *sw, *so, *se;    // south (pBelow)
					uchar *pDst;
					pAbove = nullptr;
					pCurr = ret.ptr<uchar>(0);
					pBelow = ret.ptr<uchar>(1);
					for (y = 1; y < ret.rows - 1; ++y) {
						pAbove = pCurr;
						pCurr = pBelow;
						pBelow = ret.ptr<uchar>(y + 1);
						pDst = marker.ptr<uchar>(y);
						no = &(pAbove[0]);
						ne = &(pAbove[1]);
						me = &(pCurr[0]);
						ea = &(pCurr[1]);
						so = &(pBelow[0]);
						se = &(pBelow[1]);
						for (x = 1; x < ret.cols - 1; ++x) {
							nw = no;
							no = ne;
							ne = &(pAbove[x + 1]);
							we = me;
							me = ea;
							ea = &(pCurr[x + 1]);
							sw = so;
							so = se;
							se = &(pBelow[x + 1]);
							int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
								(*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
								(*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
								(*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
							int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
							int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
							int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);
							if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
								pDst[x] = 1;
						}
					}
					ret &= ~marker;
				}
				cv::absdiff(ret, prev_img, diff);
				prev_img = ret.clone();
			} while (cv::countNonZero(diff) > 0);
			ret *= 255;
			for (int y = 1; y < ret.rows - 1; y++) {
				for (int x = 1; x < ret.cols - 1; x++) {
					bool up = ret.at<uchar>(y - 1, x) == 255;
					bool down = ret.at<uchar>(y + 1, x) == 255;
					bool left = ret.at<uchar>(y, x - 1) == 255;
					bool right = ret.at<uchar>(y, x + 1) == 255;
					int c = (up&right) + (up&left) + (down&left) + (down&right);
					if (c==1) {
						ret.at<uchar>(y, x) = 0;
					}
				}
			}
			return ret;
		}
		/**
		*	@brief 이진 이미지에서 다중 연결점을 제거 합니다.
		*	@param image 입력 이미지
		*	@warning 입력 이미지는 반드시 이진 이미지여야 합니다.
		*	@return 연결 픽셀이 제거된 이미지
		*	@remark
		*		<img src="https://i.imgur.com/cOP19G0.png" width="640">
		*/
		static cv::Mat PixelDisconnect(cv::Mat image) {
			if (image.channels() != 1 || image.depth() != 0) {
				ISPRING_VERIFY("This is not binary image");
			}
			if (image.rows <= 3 || image.cols <= 3) {
				ISPRING_VERIFY("Image size is too small");
			}
			cv::Mat img = image.clone();
			cv::Rect img_rect(0, 0, img.rows, img.cols);
			for (int y = 0; y < img.rows; y++) {
				for (int x = 0; x < img.cols; x++) {
					if (img.at<uchar>(y, x) == 255) {
						int link = 0;
						for (int ym = -1; ym <= 1; ym++)
							for (int xm = -1; xm <= 1; xm++)
								if (img_rect.contains(cv::Point(y + ym, x + xm)))
									link += img.at<uchar>(y + ym, x + xm) / 255;
						if (link > 4) {
							img.at<uchar>(y, x) = 0;
						}
					}
				}
			}
			return img;
		}
		/**
		*	@brief 연결점이 제거된 이진 이미지에서 픽셀 체인을 찾습니다.
		*	@param image 입력 이미지
		*	@warning 입력 이미지는 반드시 연결점이 제거된 이진 이미지여야 합니다.\n
		*	연결점이 제거 되지 않아도 동작은 하지만 이후에 동작은 보장하지 않습니다.
		*	@return 픽셀 체인
		*	@remark
		*		<img src="https://i.imgur.com/L4iIMZa.png" width="640">
		*/
		static std::vector<std::vector<cv::Point>> PixelChain(cv::Mat image) {
			auto isEndPoint = [](cv::Mat img, int x, int y) ->bool {
				int c = 0;
				for (int my = -1; my <= 1; my++) {
					for (int mx = -1; mx <= 1; mx++) {
						c += img.at<uchar>(y + my, x + mx) / 255;
					}
				}
				return c == 2;
			};
			std::vector<std::vector<cv::Point>> chain;
			cv::Mat img = image.clone();
			cv::Rect img_rect(0, 0, img.cols - 1, img.rows - 1);
			for (int y = 1; y < img.rows - 1; y++) {
				for (int x = 1; x < img.cols - 1; x++) {
					if (img.at<uchar>(y, x) == 255) {
						std::deque<cv::Point> pc;
						pc.push_back(cv::Point(x, y));
						img.at<uchar>(y, x) = 0;
						for (int k = 0; k < 2; k++) {
							bool finish = false;
							while (finish == false) {
								finish = true;
								cv::Point top = k == 0 ? pc.back() : pc.front();
								for (int my = -1; my <= 1; my++) {
									for (int mx = -1; mx <= 1; mx++) {
										if (img_rect.contains(cv::Point(top.x + mx, top.y + my)) && img.at<uchar>(top.y + my, top.x + mx) == 255) {
											if (k == 0) {
												pc.push_back(cv::Point(top.x + mx, top.y + my));
											} else {
												pc.push_front(cv::Point(top.x + mx, top.y + my));
											}
											img.at<uchar>(top.y + my, top.x + mx) = 0;
											finish = false;
											goto L;
										}
									}
								}
							L: {}
							}
						}
						chain.push_back(std::vector<cv::Point>(pc.begin(),pc.end()));
					}
				}
			}
			return chain;
		}
		/**
		*	@brief 인덱스에 따라 다른 색상을 반환 합니다. \n
		*	색 파레트는 https://xkcd.com/color/rgb/ 를 사용합니다. \n
		*	색은 907가지가 있습니다. 밝은 색들만 존재합니다.\n
		*	색을 구분할때 더이상 랜덤색상을 사용하지 마십시오.
		*	@param index 색상 인덱스
		*	@return RGB색상
		*	@remark
		*		<img src="https://i.imgur.com/s0jntOf.png" width="640">
		*/
		static cv::Scalar GetRGB(size_t index) {
			return ispring_3rdparty::color_map[index%ispring_3rdparty::color_map.size()];
		}
		/**
		*	@brief Canny Edge를 수행합니다.
		*	@details 이 함수는 일반적인 이미지에서 적절한 문턱치 값으로 cv::Canny 를 수행합니다.
		*	@param image 입력 이미지
		*	@warning 현재 이진 이미지에서는 제대로 동작하지 않습니다.
		*	@return 윤곽 이미지
		*	@remark
		*		<img src="https://i.imgur.com/M0LxN7q.jpg" width="640" >
		*/
		static cv::Mat AutoCanny(cv::Mat image) {
			cv::Mat gray, edge, dummy;
			if (image.channels() == 3) {
				cv::cvtColor(image, gray, CV_BGR2GRAY);
			}
			else {
				gray = image;
			}
			double high_th = cv::threshold(gray, dummy, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
			double low_th = high_th / 2.;
			cv::Canny(gray, edge, low_th, high_th);
			return edge; 
		}
		/**
		*	@brief 적절한 크기로 화면에 이미지를 출력합니다.
		*	@details 이미지가 너무 크거나 작을때도 항상 화면에 적절한 크기로 출력합니다.
		*	@param image 입력 이미지
		*	@param stop 출력후 대기할지 결정합니다. 
		*	@param full 전체 화면으로 출력할지 결정합니다.
		*	@remark
		*		<img src="https://i.imgur.com/ga1KZYi.jpg" width="640" >
		*/
		static void DisplayImage(cv::Mat image, bool stop = true, bool full = false) {
#ifdef ISPRING_WINDOWS
			cv::Mat rsz;
			cv::Size sz;
			bool op = (image.cols < image.rows);
			double sq = (double)image.cols / image.rows;
			int tmp = GetSystemMetrics(op) / (full ? 1 : 2);
			(op ? sz.height : sz.width) = tmp;
			(op ? sz.width : sz.height) = static_cast<int>(tmp * (op ? sq*sq : 1) / sq);
			cv::resize(image, rsz, sz,0,0,cv::INTER_CUBIC);
			cv::Mat out;
			rsz.convertTo(out, CV_8UC3);
			cv::imshow("image", out);
			if (stop == true) {
				cv::waitKey();
				cv::destroyAllWindows();
			}
			else {
				if (cv::waitKey(1) == 27) {
					exit(1);
				}
			}
#endif
		}
		/**
		*	@brief 윈도우의 기본 사진 앱에 Mat영상을 출력합니다.
		*	@param image 입력 이미지
		*	@param async 출력후 대기할지 결정합니다. false를 주면 사진 뷰어 앱이 꺼질때까지 대기합니다.
		*	@remark
		*		<img src="https://i.imgur.com/k56TpJF.jpg" width="640" >
		*/
		static void DisplayImage2(cv::Mat image, bool async = false) {
#ifdef ISPRING_WINDOWS
			char _path[MAX_PATH + 1];
			GetTempPathA(MAX_PATH, _path);
			std::string path = _path + std::string("ispringDisplayImage2/") + std::to_string(clock()) + ".png";
			ispring::File::DirectoryErase(_path + std::string("ispringDisplayImage2/"));
			ispring::File::DirectoryMake(_path + std::string("ispringDisplayImage2/"));
			cv::imwrite(path, image);
			if (async) {
				ShellExecuteA(NULL, "open", path.c_str(), NULL, NULL, SW_SHOW);
			} else {
				system(path.c_str());
				DeleteFileA(path.c_str());
			}
#endif
		}
		/**
		*	@brief 이미지를 확대하거나 축소 합니다.
		*	@param image 입력 이미지
		*	@param ratio 확대 또는 축소할 비율. 이 값이 1.0F 미만이면 작게 축소되며, 1.0F 이상이면 크게 확대됨.
		*	@return 확대 또는 축소된 이미지
		*	@remark
		*		<img src="https://i.imgur.com/CZVTaZh.jpg" width="1280" >
		*/
		static cv::Mat Zoom(cv::Mat image, double ratio) {
			if (ratio == 1.0F) {
				return image.clone();
			} else if (ratio < 1.0F) {
				cv::Size zsize(static_cast<int>(ratio*image.cols), static_cast<int>(ratio*image.rows));
				cv::Mat zoomed = cv::Mat::zeros(image.size(), CV_8UC3);
				cv::Mat small_img;
				cv::resize(image, small_img, zsize,0,0,cv::INTER_CUBIC);
				int left_padding = (image.cols - small_img.cols) / 2;
				int top_padding = (image.rows - small_img.rows) / 2;
				small_img.copyTo(zoomed(cv::Rect(left_padding, top_padding, small_img.cols, small_img.rows)));
				return zoomed;
			} else {	//ratio>1.0F
				cv::Size zsize(static_cast<int>(ratio*image.cols), static_cast<int>(ratio*image.rows));
				cv::Mat zoomed = cv::Mat::zeros(image.size(), CV_8UC3);
				cv::Mat large_img;
				cv::resize(image, large_img, zsize,0,0, cv::INTER_CUBIC);
				int left_margin = (large_img.cols - image.cols) / 2;
				int top_margin = (large_img.rows - image.rows) / 2;
				zoomed = large_img(cv::Rect(left_margin, top_margin, large_img.cols - left_margin * 2, large_img.rows - top_margin * 2));
				return zoomed(cv::Rect(0, 0, image.cols, image.rows));
			}
		}
		/**
		*	@brief 이미지를 원래 크기에 맞게 회전 시킵니다.
		*	@param image 입력 이미지
		*	@param degree 회전할 각도 CW(시계방향)
		*	@param out 회전 중심점을 반환함.
		*	@return 회전된 이미지
		*	@remark
		*		<img src="https://i.imgur.com/5uHt0Eb.jpg" width="640" >
		*/
		static cv::Mat ImageRotateOuter(const cv::Mat image, double degree, cv::Point* out = nullptr) {
			cv::Point2d base(image.cols / 2.0, image.rows / 2.0);
			cv::Mat rot = cv::getRotationMatrix2D(base, degree, 1.0);
			cv::Rect bbox = cv::RotatedRect(base, image.size(),(float) degree).boundingRect();
			rot.at<double>(0, 2) += bbox.width / 2.0 - base.x;
			rot.at<double>(1, 2) += bbox.height / 2.0 - base.y;
			if (out != nullptr) {
				out->x = static_cast<int>(bbox.width / 2.0 - base.x);
				out->y = static_cast<int>( bbox.height / 2.0 - base.y);
			}
			cv::Mat dst;
			cv::warpAffine(image, dst, rot, bbox.size());
			return std::move(dst);
		}
		/**
		*	@brief 이미지를 원래 크기에 맞게 회전 시킵니다.
		*	@param image 입력 이미지
		*	@param degree 회전할 각도 CW(시계방향)
		*	@param base 기본 회전 중심점. 기본 매개변수를 사용하면 이미지의 중점으로 회전 합니다.
		*	@return 회전된 이미지
		*	@remark
		*		<img src="https://i.imgur.com/evXPXAt.jpg" width="640" >
		*/
		static cv::Mat ImageRotateInner(const cv::Mat image, double degree, cv::Point2d base=cv::Point2d(-1,-1)) {
			if (base.x == -1 && base.y == -1) {
				base.x = image.cols / 2;
				base.y = image.rows / 2;
			}
			cv::Mat dst = image.clone();
			cv::Mat rot = cv::getRotationMatrix2D(base, degree, 1.0);
			cv::warpAffine(image, dst, rot, image.size());
			return std::move(dst);
		}
		/**
		*	@brief 이미지를 지정한 width와 height에 비율의 변화 없이 피팅 시킵니다.
		*	@param _img 입력 이미지
		*	@param w 가로크기
		*	@param h 세로크기
		*	@param pad_color 패딩이 생길경우 채울 Gray 색상
		*	@return 피팅된 이미지
		*	@remark
		*		<img src="https://i.imgur.com/wNt5dVY.jpg" width="640" >
		*/
		static cv::Mat FitImage(cv::Mat _img, int w, int h,uchar pad_color=144) {
			cv::Mat img = _img.clone();
			if (img.channels() == 1) {
				cv::cvtColor(img, img, CV_GRAY2BGR);
			}
			cv::Mat ret = cv::Mat::ones(h, w, CV_8UC1);
			ret *= pad_color;
			cv::cvtColor(ret, ret, CV_GRAY2BGR);
			if (img.cols != w || img.rows != h) {
				int newWidth = static_cast<int>(img.cols*(h / (double)img.rows));
				int newHeight = h;
				if (newWidth > w) {
					newWidth = w;
					newHeight = static_cast<int>(img.rows*(w / (double)img.cols));
				}
				cv::resize(img, img, cv::Size(newWidth, newHeight), 0, 0, CV_INTER_NN);
			}
			img.copyTo(ret(cv::Rect((w-img.cols)/2,(h-img.rows)/2,img.cols,img.rows)));
			return ret;
		}
		/**
		*	@brief 이미지들을 하나의 이미지로 합칩니다.
		*	@details 이미지가 2장이면 가로,세로 비율을 보고 위아래로 붙일지 좌우로 붙일지 결정합니다.\n
		*			그 이외의 경우는 NxN 매트릭스에 순서대로 이미지를 쌓습니다. 모자란 이미지는 지정한 Pad 색으로 칠해 집니다.\n
		*			매트릭스의 기본 크기는 첫번째 이미지의 크기를 따릅니다.\n
		*			자세한 내용은 예제 이미지를 참조하십시오
		*	@param imgs 입력 이미지들
		*	@param pad_color 패딩이 생길경우 채울 Gray 색상
		*	@return 합쳐진 이미지
		*	@remark
		*		<p>&nbsp;</p><img src="https://i.imgur.com/gYZL4Hc.jpg" width="640" >
		*		<p>&nbsp;</p><img src="https://i.imgur.com/wBqAbWh.jpg" width="640" >
		*		<p>&nbsp;</p><img src="https://i.imgur.com/jfuCqjN.jpg" width="640" >
		*		<p>&nbsp;</p><img src="https://i.imgur.com/mmygjDv.jpg" width="640" >
		*		<p>&nbsp;</p><img src="https://i.imgur.com/Byg6XvY.jpg" width="640" >
		*		<p>&nbsp;</p><img src="https://i.imgur.com/XrTis3c.jpg" width="640" >
		*/
		static cv::Mat GlueImage(std::vector<cv::Mat> imgs,uchar pad_color=144) {
			if (imgs.size() == 0) {
				return cv::Mat();
			}
			if (imgs.size() == 1) {
				return imgs.front();
			}
			cv::Size stdsz = imgs.front().size();
			cv::Mat dummy = cv::Mat::ones(stdsz, CV_8UC1)*pad_color;
			cv::cvtColor(dummy, dummy, CV_GRAY2BGR);
			std::vector<cv::Mat> nimg;
			for (auto&img : imgs) {
				cv::Mat fitimg = FitImage(img, stdsz.width, stdsz.height);
				nimg.push_back(fitimg);
			}
			cv::Mat glue;
			if (nimg.size() == 2) {
				if ((stdsz.width*9)/16 > stdsz.height) {
					cv::vconcat(nimg[0], nimg[1], glue);
				} else {
					cv::hconcat(nimg[0], nimg[1], glue);
				}
			} else {
				int target_size = 2;
				while (static_cast<int>(nimg.size()) > target_size*target_size) {
					target_size++;
				}
				for (int i = static_cast<int>(nimg.size()); i < target_size*target_size; i++) {
					nimg.push_back(dummy);
				}
				std::vector<cv::Mat> lines;
				for (int i = 0; i < target_size; i++) {
					for (int j = 0; j < target_size-1; j++) {
						cv::hconcat(nimg[i*target_size + j], nimg[i*target_size + j + 1], nimg[i*target_size + j + 1]);
						nimg[i*target_size + j].release();
					}
					lines.push_back(nimg[i*target_size + target_size - 1]);
				}
				for (int i = 0; i < static_cast<int>(lines.size()) - 1; i++) {
					cv::vconcat(lines[i], lines[i + 1], lines[i + 1]);
					lines[i].release();
				}
				glue = lines.back();
				//4,9,16,25...n^2
			}
			return glue;
		}
#if defined(ISPRING_WINDOWS) || defined(DOXYGEN)
		/**
		*	@brief HBITMAP을 cv::Mat 으로 변환 합니다.
		*	@details 이 함수는 Windows 전용 입니다.
		*	@param hbmp HBITMAP 이미지
		*	@return cv::Mat
		*/
		cv::Mat HBITMAP2cvMat(HBITMAP hbmp) {
			static HWND hwnd = nullptr;
			if (hwnd == nullptr) {
				hwnd = ispring::OS::GetHWND();
			}
			cv::Mat img;
			HDC hdc = GetDC(hwnd);
			BITMAPINFO bmi;
			BITMAPINFOHEADER* bmih = &(bmi.bmiHeader);
			ZeroMemory(bmih, sizeof(BITMAPINFOHEADER));
			bmih->biSize = sizeof(BITMAPINFOHEADER);
			if (GetDIBits(hdc, hbmp, 0, 0, NULL, &bmi, DIB_RGB_COLORS)) {
				int height = (bmih->biHeight > 0) ? bmih->biHeight : -bmih->biHeight;
				img = cv::Mat(height, bmih->biWidth, CV_8UC3);

				bmih->biBitCount = 24;
				bmih->biCompression = BI_RGB;
				GetDIBits(hdc, hbmp, 0, height, img.data, &bmi, DIB_RGB_COLORS);
			}
			ReleaseDC(NULL, hdc);
			cv::flip(img, img, 0);
			return img;
		}
#endif
	};
}
#endif
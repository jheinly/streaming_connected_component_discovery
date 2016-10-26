#ifndef OCV_LIB_UTILS_H
#define OCV_LIB_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>

cv::Mat PfromRt(const cv::Mat & R, const cv::Mat & t);
void computeCofactorsMat(const cv::Mat & M, cv::Mat & cofact);

void projectOnSO3(const cv::Mat& src, cv::Mat& dest);

void ppv(const cv::Mat& t, cv::Mat & res);

bool checkForForwardMotion(const cv::Mat& R, const cv::Mat& t);

double getSquaredRerojectionError(const cv::Mat & pt_3d, const cv::Mat & pt_2d,
                                  const cv::Mat & K, const cv::Mat & dist, const cv::Mat & R, const cv::Mat & t);
size_t checkRerojectionError(const cv::Mat & pts_3d, const cv::Mat & pts_2d,
                           const cv::Mat & K, const cv::Mat & dist, const cv::Mat & R, const cv::Mat & t,
                           std::vector<bool> & status,
                           double threshold);

void saveMatForMatlab(const cv::Mat & mat, const std::string & fn);

double getMeanSquaredRerojectionError(const std::vector<cv::Mat> & ps,
                                      const std::vector<cv::Mat> & Ks,
                                      const std::vector<cv::Mat> & dists,
                                      const std::vector<cv::Mat> & Rs,
                                      const std::vector<cv::Mat> & ts,
                                      const cv::Mat & M);

namespace ocv_lib {


void getCameraLocation(const cv::Mat& R, const cv::Mat& t, cv::Mat& C);
void getCameraLocation(const cv::Mat& P, cv::Mat& C);
  ///It expects double type!
  ///It allocates pts_2d if needed as a 2xN matrix, if it is pre_allocated it can be 3xN or 2xN;

  void projectPoints(const cv::Mat & pts_3d, const cv::Mat & camera_matrix, const cv::Mat & dist, const cv::Mat & R, const cv::Mat & t,
                      cv::Mat & pts_2d);
  void undistortPoints(const cv::Mat & pts_dist, const cv::Mat & camera_matrix, const cv::Mat & dist, cv::Mat & pts_undist);

  void getReprojectionError(const cv::Mat & pts_3d, const cv::Mat & pts_2d, const cv::Mat & camera_matrix, const cv::Mat & dist, const cv::Mat & R, const cv::Mat & t,  double * err_);

}
#endif //OCV_LIB_UTILS_H

#ifndef OCV_LIB_TRIANGULATION_H
#define OCV_LIB_TRIANGULATION_H

#include <opencv2/opencv.hpp>
#include "OCV_utils.h"

#include <string>
#include <fstream>
#include <math.h>
#include <vector>
using std::vector;

/**
 * Compute the cosinus of the apical angle from pt
 * @param C1 first camera location 
 * @param C2 second camera location
 * @param pt 3D points
 */
double GetCosApicalAngle(const cv::Mat & C1, const cv::Mat & C2, const cv::Mat & pt);

double GetApicalAngle(const cv::Mat & C1, const cv::Mat & C2, const cv::Mat & pt);

bool checkCherality(const cv::Mat & R, const cv::Mat & t, const cv::Mat & M);

void reconstruct3dPoint(const cv::Mat & m1, const cv::Mat & m2, const cv::Mat & P2, cv::Mat &  M);

double reconstruct3dPoint(const vector<cv::Mat> & ps, 
                          const vector<cv::Mat> & Ks, 
                          const vector<cv::Mat> & dist, 
                          const vector<cv::Mat> & Rs, 
                          const vector<cv::Mat> & ts, 
                          cv::Mat & M);

size_t cheralityCheck(const cv::Mat & R, const cv::Mat & t, 
		      const cv::Mat & m1, const cv::Mat & m2,	
		      std::vector<bool> & status);

size_t getNumberOfStableTriangulationBasedOnRoundness(const cv::Mat & p1, const cv::Mat & p2,
                                                const cv::Mat & K1, const cv::Mat & K2,
                                                const cv::Mat & dist1, const cv::Mat & dist2,
                                                const cv::Mat & P, std::vector<bool> & status, 
                                                double threshold);

double computeBestRoundness(const std::vector<cv::Mat > &pts,
                            const std::vector<cv::Mat > &Ks,
                            const std::vector<cv::Mat > &dists,
                            const std::vector<cv::Mat > & Rs, 
                            const std::vector<cv::Mat > & ts,
                            const vector<size_t> & imageIndexes);

double computeBestAA(const cv::Mat & M,
                      const vector<cv::Mat > & Rs, 
                      const vector<cv::Mat > & ts,
                      const vector<int> & imageIndexes);

double getAverageApicalAngleGetStability(const cv::Mat & p1, const cv::Mat & p2,
                                          const cv::Mat & K1, const cv::Mat & K2,
                                          const cv::Mat & dist1, const cv::Mat & dist2,
                                          const cv::Mat & P, std::vector<bool> & status, 
                                          const double min_angle_stability_threshold,
                                          size_t & numOfStablePoint);

#endif //OCV_LIB_TRIANGULATION_H

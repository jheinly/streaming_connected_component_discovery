#ifndef OCV_LIB_ESSENTIAL_MAT_H
#define OCV_LIB_ESSENTIAL_MAT_H

#include <vector>

#include <opencv2/opencv.hpp>

#include <estimator/v3d_code/v3d_poseutilities.h>
#include <estimator/v3d_code/v3d_linear.h>
#include <estimator/v3d_code/v3d_ocv.h>
#include <estimator/v3d_code/OCV_utils.h>
#include <estimator/v3d_code/triangulation.h>


void computeEssentialFromRT(const cv::Mat & R,
                            const cv::Mat & t,
                            cv::Mat E);

void decomposeEssentialMat(const cv::Mat & E, 
						   cv::Mat & R1, 
						   cv::Mat & R2, 
						   cv::Mat & t1, 
						   cv::Mat & t2);

double getEssentialRatio2(cv::Mat & imagePoint_src, 
                         cv::Mat & imagePoint_tgt, 
                         cv::Mat & K_src, 
                         cv::Mat & K_tgt);

size_t selectCorrectRtUsingCherality(const cv::Mat & R1,
				     const cv::Mat & R2,
				     const cv::Mat & t1,
				     const cv::Mat & t2,
				     const cv::Mat & m1, 
				     const cv::Mat & m2, 
				     std::vector<bool> & status,
				     cv::Mat & R,
				     cv::Mat & t);

size_t refineRTfromEssentialusingDepthConstraint( const cv::Mat & pts_src, const cv::Mat & pts_tgt,
                                                 const cv::Mat & K_src, const cv::Mat & K_tgt,
                                                 cv::Mat & E,
												                         cv::Mat & R, cv::Mat & t,
												                         std::vector<bool> & status);

size_t refineRTfromEssentialUsingPointTriangulation( const cv::Mat & pts_src, const cv::Mat & pts_tgt,
                                                    const cv::Mat & K_src, const cv::Mat & K_tgt,
                                                     cv::Mat & E,
												                             cv::Mat & R, cv::Mat & t,
												                              std::vector<bool> & status);

double getEssentialRatio(cv::Mat & imagePoint_src, 
                         cv::Mat & imagePoint_tgt, 
                         cv::Mat & K_src, 
                         cv::Mat & K_tgt);
#endif // OCV_LIB_ESSENTIAL_MAT_H

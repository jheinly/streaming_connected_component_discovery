#ifndef V3D_OCV_H
#define V3D_OCV_H

#include <opencv2/opencv.hpp>
#include <estimator/v3d_code/v3d_linear.h>

namespace ocv_lib {
  void translate(const V3D::Matrix3x3d & v3d_mat, cv::Mat & ocv_mat);

void translate(const cv::Mat & ocv_mat, V3D::Matrix3x3d & v3d_mat);


void translate(const V3D::Vector3d & v3d_vec, cv::Mat & ocv_vec);

void translate(const cv::Mat & ocv_vec, V3D::Vector3d & v3d_vec);
  ///It expects double type!
  ///It allocates pts_2d if needed as a 2xN matrix, if it is pre_allocated it can be 3xN or 2xN;

}

#endif

#include <estimator/v3d_code/v3d_ocv.h>

void ocv_lib::translate(const V3D::Matrix3x3d & v3d_mat, cv::Mat & ocv_mat) {
  ocv_mat.create(3,3,CV_64FC1);
  for (int i=0; i < 3; ++i)
    for (int j=0; j < 3; ++j)
        ocv_mat.at<double>(i,j) = v3d_mat[i][j];
 }

void ocv_lib::translate(const cv::Mat & ocv_mat, V3D::Matrix3x3d & v3d_mat) {
  for (int i=0; i < 3; ++i)
    for (int j=0; j < 3; ++j)
        v3d_mat[i][j] = ocv_mat.at<double>(i,j);
}


void ocv_lib::translate(const V3D::Vector3d & v3d_vec, cv::Mat & ocv_vec) {
  ocv_vec.create(3,1,CV_64FC1); 
  for (int i=0; i < 3; ++i)
    ocv_vec.at<double>(i,0) = v3d_vec[i];
 }

void ocv_lib::translate(const cv::Mat & ocv_vec, V3D::Vector3d & v3d_vec) {
  for (int i=0; i < 3; ++i)
    v3d_vec[i] = ocv_vec.at<double>(i);
 }

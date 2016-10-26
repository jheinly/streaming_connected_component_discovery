#include <estimator/v3d_code/Matcher.h>

/* Includes for cuda ransac */
//#include "cuda_ransac_driver.h"
//#include "ransac_conf.h"
//#include "ransac_params.h"
//#include "cuda/cuda_F7p_fit.h"
//#include "cuda/cuda_F_eval.h"

//#include "cuda_F7p_normalize.h"

#include <iostream>
using std::cerr;
using std::endl;

#include <cmath>
using std::min;

//#include <V3D/Features/Sift.h>
//#include <V3D/Features/SiftSupport.h>
//#include <V3D/Features/HarrisBrief.h>
//#include <V3D/Features/HarrisBriefSupport.h>

//#include <cuda_runtime_api.h>

//V3D_CODE_FILES

//size_t sfmData::matcher::ransacF(featInformation* im1, 
//								featInformation* im2, 
//								const std::vector<std::pair<int, int> >& matches,      
//								std::vector<std::pair<int, int> >& inliers_matches,
//								FundMatrixEstimator& F_Estimator,
//								MatcherContainer & container) 
//{
//
////	std::vector<std::pair<int, int> > matches_return;
//
//	size_t num_inliers = 0;
//	const size_t num_putatives = matches.size();
//	BIAS::Matrix3x3<double> F; F.SetZero();
//	BIAS::Matrix3x3<double> solution; solution.SetZero();
//	BIAS::Matrix3x3<double> K1; K1.SetZero();
//	BIAS::Matrix3x3<double> K2; K2.SetZero();
//	
//	// original points
//	container.pts1.resize(num_putatives);
//	container.pts2.resize(num_putatives);
//	
//	// normalized points
//	container.npts1.resize(num_putatives);
//	container.npts2.resize(num_putatives);
//	
//	container.inliers.clear();
//	container.inliers.reserve(num_putatives);
//	
//	// temporary storage
//	BIAS::Vector3<double> pt1(0.0, 0.0, 1.0);
//	BIAS::Vector3<double> pt2(0.0, 0.0, 1.0);
//	
//	if (num_putatives >= 7)
//	{
//		float (*loc1)[5] = im1->getFeature()->getLocationData(); //check if we call a men copy
//		float (*loc2)[5] = im2->getFeature()->getLocationData();
//		for (unsigned int matchIdx = 0; matchIdx < num_putatives; ++matchIdx)
//		{
//			size_t idx1 = matches[matchIdx].first;
//			size_t idx2 = matches[matchIdx].second;
//			
//			pt1.Set(loc1[idx1][0], loc1[idx1][1], 1.0);
//			pt2.Set(loc2[idx2][0], loc2[idx2][1], 1.0);
//			
//			container.pts1[matchIdx] = pt1;
//			container.pts2[matchIdx] = pt2;
//		}
//		// normalize data
//		normalizePoints(container.pts1, container.npts1, K1);
//		normalizePoints(container.pts2, container.npts2, K2);
//
//		// initialize ransac estimator with current data parameters
//		F_Estimator.initData(container.npts1, container.npts2);
//		F_Estimator.initNormalization(container.pts1, container.pts2, K1,K2);
//		// compute solution
//		num_inliers = F_Estimator.solveMaster(solution, container.inliers);
//
//		F_Estimator.deallocateData();
//		
//		inliers_matches.resize(num_inliers);
//		if (num_inliers) {
//			for(size_t j = 0, pos=0; j < num_putatives; ++j)
//				if(container.inliers[j]) 
//					inliers_matches[pos++] = matches[j];
//		}
//		
//	}
//		
//	return num_inliers;
//}

//size_t sfmData::matcher::ransacF_ocv(const V3D::Features& feat1, 
//  const V3D::Features& feat2, 
//  const std::vector<std::pair<int, int> >& matches,      
//  std::vector<std::pair<int, int> >& inliers_matches,
//  ocv_lib::ARRSAC_fundamental& F_Estimator,
//  MatcherContainer & container,
//  cv::Mat & solution) 
//{
//
//  //	std::vector<std::pair<int, int> > matches_return;
//
//  size_t num_inliers = 0;
//  const size_t num_putatives = matches.size();
//  //cv::Mat solution;
//
//  // original points
//  cv::Mat pts1(3, static_cast<int>(num_putatives), CV_64F);
//  cv::Mat pts2(3, static_cast<int>(num_putatives), CV_64F);
//
//  container.inliers.clear();
//  container.inliers.reserve(num_putatives);
//
//  if (num_putatives >= 7)
//  {
//    for (size_t matchIdx = 0; matchIdx < num_putatives; ++matchIdx)
//    {
//      int idx1 = matches[matchIdx].first;
//      int idx2 = matches[matchIdx].second;
//
//      pts1.at<double>(0, static_cast<int>(matchIdx)) = feat1.genericKeypoint(idx1).x;
//      pts1.at<double>(1, static_cast<int>(matchIdx)) = feat1.genericKeypoint(idx1).y;
//      pts1.at<double>(2, static_cast<int>(matchIdx)) = 1.0;
//
//      pts2.at<double>(0, static_cast<int>(matchIdx)) = feat2.genericKeypoint(idx2).x;
//      pts2.at<double>(1, static_cast<int>(matchIdx)) = feat2.genericKeypoint(idx2).y;
//      pts2.at<double>(2, static_cast<int>(matchIdx)) = 1.0;
//    }
//
//
//    // initialize ransac estimator with current data parameters
//    F_Estimator.initData(pts1, pts2);
//    // compute solution
//    num_inliers = F_Estimator.solveMaster(solution, container.inliers);
//
//    inliers_matches.resize(num_inliers);
//    if (num_inliers) {
//      for(size_t j = 0, pos=0; j < num_putatives; ++j)
//        if(container.inliers[j]) 
//          inliers_matches[pos++] = matches[j];
//    }
//
//  }
//
//  return num_inliers;
//}
//
//size_t sfmData::matcher::ransacH_ocv(const V3D::Features& feat1, 
//  const V3D::Features& feat2, 
//  const std::vector<std::pair<int, int> >& matches,      
//  std::vector<std::pair<int, int> >& inliers_matches,
//  ocv_lib::ARRSAC_homography& H_Estimator,
//  MatcherContainer & container) 
//{
//
//  //	std::vector<std::pair<int, int> > matches_return;
//
//  size_t num_inliers = 0;
//  const size_t num_putatives = matches.size();
//  cv::Mat solution;
//
//  // original points
//  cv::Mat pts1(3, static_cast<int>(num_putatives), CV_64F);
//  cv::Mat pts2(3, static_cast<int>(num_putatives), CV_64F);
//
//  container.inliers.clear();
//  container.inliers.reserve(num_putatives);
//
//  if (num_putatives >= 4)
//  {
//    for (size_t matchIdx = 0; matchIdx < num_putatives; ++matchIdx)
//    {
//      int idx1 = matches[matchIdx].first;
//      int idx2 = matches[matchIdx].second;
//
//      pts1.at<double>(0, static_cast<int>(matchIdx)) = feat1.genericKeypoint(idx1).x;
//      pts1.at<double>(1, static_cast<int>(matchIdx)) = feat1.genericKeypoint(idx1).y;
//      pts1.at<double>(2, static_cast<int>(matchIdx)) = 1.0;
//
//      pts2.at<double>(0, static_cast<int>(matchIdx)) = feat2.genericKeypoint(idx2).x;
//      pts2.at<double>(1, static_cast<int>(matchIdx)) = feat2.genericKeypoint(idx2).y;
//      pts2.at<double>(2, static_cast<int>(matchIdx)) = 1.0;
//    }
//
//
//    // initialize ransac estimator with current data parameters
//    H_Estimator.initData(pts1, pts2);
//    // compute solution
//    num_inliers = H_Estimator.solveMaster(solution, container.inliers);
//
//    inliers_matches.resize(num_inliers);
//    if (num_inliers) {
//      for(size_t j = 0, pos=0; j < num_putatives; ++j)
//        if(container.inliers[j]) 
//          inliers_matches[pos++] = matches[j];
//    }
//
//  }
//
//  return num_inliers;
//}
//
//size_t sfmData::matcher::ransacH_ocv(const V3D::Features& feat1, 
//  const V3D::Features& feat2, 
//  const std::vector<std::pair<int, int> >& matches,      
//  std::vector<std::pair<int, int> >& inliers_matches,
//  ocv_lib::ARRSAC_homography& H_Estimator,
//  MatcherContainer & container,
//  cv::Mat& solution) 
//{
//
//  //	std::vector<std::pair<int, int> > matches_return;
//
//  size_t num_inliers = 0;
//  const size_t num_putatives = matches.size();
//  //cv::Mat solution;
//
//  // original points
//  cv::Mat pts1(3, static_cast<int>(num_putatives), CV_64F);
//  cv::Mat pts2(3, static_cast<int>(num_putatives), CV_64F);
//
//  container.inliers.clear();
//  container.inliers.reserve(num_putatives);
//
//  if (num_putatives >= 4)
//  {
//    for (size_t matchIdx = 0; matchIdx < num_putatives; ++matchIdx)
//    {
//      int idx1 = matches[matchIdx].first;
//      int idx2 = matches[matchIdx].second;
//
//      pts1.at<double>(0, static_cast<int>(matchIdx)) = feat1.genericKeypoint(idx1).x;
//      pts1.at<double>(1, static_cast<int>(matchIdx)) = feat1.genericKeypoint(idx1).y;
//      pts1.at<double>(2, static_cast<int>(matchIdx)) = 1.0;
//
//      pts2.at<double>(0, static_cast<int>(matchIdx)) = feat2.genericKeypoint(idx2).x;
//      pts2.at<double>(1, static_cast<int>(matchIdx)) = feat2.genericKeypoint(idx2).y;
//      pts2.at<double>(2, static_cast<int>(matchIdx)) = 1.0;
//    }
//
//
//    // initialize ransac estimator with current data parameters
//    H_Estimator.initData(pts1, pts2);
//    // compute solution
//    num_inliers = H_Estimator.solveMaster(solution, container.inliers);
//
//    inliers_matches.resize(num_inliers);
//    if (num_inliers) {
//      for(size_t j = 0, pos=0; j < num_putatives; ++j)
//        if(container.inliers[j]) 
//          inliers_matches[pos++] = matches[j];
//    }
//
//  }
//
//  return num_inliers;
//}

size_t sfmData::matcher::ransacE_ocv(const V3D::Features& feat1, 
  const V3D::Features& feat2,  
  const double * k1,
  const double * k2,
  const std::vector<std::pair<int, int> >& matches,      
  std::vector<std::pair<int, int> >& inliers_matches,
  ocv_lib::ARRSAC_essential& E_Estimator_ocv,
  MatcherContainer & container,
  cv::Mat & solution,
  bool shouldCheckForLineDegeneracy) 
{
  size_t num_inliers = 0;
  const size_t num_putatives = matches.size();
  double dist_[4], K1_[9], K2_[9];
  cv::Mat K1(3,3,CV_64FC1,K1_); 
  K1 = cv::Mat::eye(3,3,CV_64FC1);  
  cv::Mat K2(3,3,CV_64FC1,K2_); //tocheck this might have questionable performances...
  K2  = cv::Mat::eye(3,3,CV_64FC1);  
  cv::Mat dist(4,1,CV_64FC1,dist_);  
  dist_[0] = dist_[1] = dist_[2] = dist_[3] = 0.0; //set to sero

  cv::Mat pts1(2, static_cast<int>(num_putatives), CV_64FC1);
  cv::Mat pts2(2, static_cast<int>(num_putatives), CV_64FC1);
  container.inliers.clear();
  container.inliers.reserve(num_putatives);

  

  // temporary storage
  if (num_putatives >= 5)
  {
    for (unsigned int matchIdx = 0; matchIdx < num_putatives; ++matchIdx)
    {
      pts1.at<double>(0,matchIdx) = feat1.genericKeypoint(matches[matchIdx].first).x; 
      pts1.at<double>(1,matchIdx) = feat1.genericKeypoint(matches[matchIdx].first).y;
      pts2.at<double>(0,matchIdx) = feat2.genericKeypoint(matches[matchIdx].second).x; 
      pts2.at<double>(1,matchIdx) = feat2.genericKeypoint(matches[matchIdx].second).y;
    }

    K1_[0] = K1_[4]= k1[0];//k1->f;
    K1_[2] = k1[1];//->px;
    K1_[5] = k1[2];//->py;
    K2_[0] = K2_[4]= k2[0];//->f;
    K2_[2] = k2[1];//->px;
    K2_[5] = k2[2];//->py;

    // initialize ransac estimator with current data parameters
    E_Estimator_ocv.initNormalization(pts1, pts2, K1, K2, dist, dist);
    // compute solution
    num_inliers = E_Estimator_ocv.solveMaster(solution, container.inliers);
    E_Estimator_ocv.clear();
    inliers_matches.resize(num_inliers);
    if (num_inliers) {
      for(size_t j = 0, pos=0; j < num_putatives; ++j)
        if(container.inliers[j]) 
          inliers_matches[pos++] = matches[j];
    }

  }

  if (shouldCheckForLineDegeneracy)
  {
    float res1 = static_cast<float>(2.0f * MAX(k1[1], k1[2]));
    float res2 = static_cast<float>(2.0f * MAX(k2[1], k2[2]));

    float thresh1 = res1 * 8.0f / 1000;
    float thresh2 = res2 * 8.0f / 1000;

    bool lineDegeneracyFound = checkForLineDegeneracy(feat1,
      feat2,
      inliers_matches,
      thresh1,
      thresh2);
    if (lineDegeneracyFound)
    {
      return 0;
    }
  }

  return num_inliers;
}

bool sfmData::matcher::checkForLineDegeneracy(const V3D::Features& feat1, 
  const V3D::Features& feat2,
  std::vector<std::pair<int, int> >& inlier_matches,
  float inlier_threshold1,
  float inlier_threshold2)
{
  const int num_inliers = static_cast<int>(inlier_matches.size());

  if (num_inliers < 3)
  {
    return false;
  }

  float confidence = 0.99f;
  float min_percent_on_line = 0.6f;
  float ransac_confidence_min_percent_on_line = min_percent_on_line - 0.15f;

  int max_num_iterations = int(logf(1 - confidence) / log(1 - powf(ransac_confidence_min_percent_on_line, 2)) + 1.0f);

  // Test the first camera
  for (int iter = 0; iter < max_num_iterations; ++iter)
  {
    int idx1 = rand() % num_inliers;
    int idx2 = idx1;
    while (idx1 == idx2)
    {
      idx2 = rand() % num_inliers;
    }

    float x1 = feat1.genericKeypoint(inlier_matches[idx1].first).x;
    float y1 = feat1.genericKeypoint(inlier_matches[idx1].first).y;

    float x2 = feat1.genericKeypoint(inlier_matches[idx2].first).x;
    float y2 = feat1.genericKeypoint(inlier_matches[idx2].first).y;

    float a = y2 - y1;
    float b = x1 - x2;
    float inv_norm = 1.0f / sqrtf(a*a + b*b);
    a *= inv_norm;
    b *= inv_norm;

    float c = -(a * x1 + b * y1);

    int num_on_line = 0;
    for (int i = 0; i < num_inliers; ++i)
    {
      float x = feat1.genericKeypoint(inlier_matches[i].first).x;
      float y = feat1.genericKeypoint(inlier_matches[i].first).y;
      float dist = fabsf(a * x + b * y + c);
      if (dist <= inlier_threshold1)
      {
        ++num_on_line;
      }
    }
    float percent_on_line = float(num_on_line) / num_inliers;
    if (percent_on_line >= min_percent_on_line)
    {
      return true;
    }
  }

  // Test the second camera
  for (int iter = 0; iter < max_num_iterations; ++iter)
  {
    int idx1 = rand() % num_inliers;
    int idx2 = idx1;
    while (idx1 == idx2)
    {
      idx2 = rand() % num_inliers;
    }

    float x1 = feat2.genericKeypoint(inlier_matches[idx1].second).x;
    float y1 = feat2.genericKeypoint(inlier_matches[idx1].second).y;

    float x2 = feat2.genericKeypoint(inlier_matches[idx2].second).x;
    float y2 = feat2.genericKeypoint(inlier_matches[idx2].second).y;

    float a = y2 - y1;
    float b = x1 - x2;
    float inv_norm = 1.0f / sqrtf(a*a + b*b);
    a *= inv_norm;
    b *= inv_norm;

    float c = -(a * x1 + b * y1);

    int num_on_line = 0;
    for (int i = 0; i < num_inliers; ++i)
    {
      float x = feat2.genericKeypoint(inlier_matches[i].second).x;
      float y = feat2.genericKeypoint(inlier_matches[i].second).y;
      float dist = fabsf(a * x + b * y + c);
      if (dist <= inlier_threshold2)
      {
        ++num_on_line;
      }
    }
    float percent_on_line = float(num_on_line) / num_inliers;
    if (percent_on_line >= min_percent_on_line)
    {
      return true;
    }
  }

  return false;
}

//size_t sfmData::matcher::computeNormalizedFloatingDesc(float*& floatDesc,
//  const V3D::Features& features,
//  size_t maxNumSIFT)
//{
//  const int SIFT_DIM = V3D::Sift::SiftDim;
//  int numFloatingDesc = min(static_cast<int>(maxNumSIFT), static_cast<int>(features.size()));
//  //floatDesc = new float[SIFT_DIM * numFloatingDesc];
//
//  for (int pointIdx = 0; pointIdx < numFloatingDesc; ++pointIdx)
//  {
//    float norm = 0.0f;
//
//    for (int descIdx = 0; descIdx < SIFT_DIM; ++descIdx)
//    {
//      float val = static_cast<float>(features.descriptor<V3D::Sift>(pointIdx).descriptor[descIdx]);
//      floatDesc[descIdx + SIFT_DIM * pointIdx] = val;
//      norm += val * val;
//    }
//
//    norm = sqrtf(norm);
//
//    for (int descIdx = 0; descIdx < SIFT_DIM; ++descIdx)
//    {
//      floatDesc[descIdx + SIFT_DIM * pointIdx] /= norm;
//    }
//  }
//
//  return numFloatingDesc;
//}


#ifdef USE_BIAS

void sfmData::matcher::normalizePoints(const std::vector< BIAS::Vector3<double> >& points,
  std::vector< BIAS::Vector3<double> >& npoints,
  BIAS::Matrix3x3<double>& K) 
{
  size_t num_points = points.size();
  npoints.resize(num_points);

  // compute mean 
  double mean_x = 0.0;
  double mean_y = 0.0;

  for (size_t i = 0; i < num_points; ++i) 
  {
    mean_x += points[i][0];
    mean_y += points[i][1];
  }
  mean_x /= static_cast<double>(num_points);
  mean_y /= static_cast<double>(num_points);

  // compute mean distance from center
  double mean_dist = 0.0;
  for (size_t i = 0; i < num_points; i++) 
  {
    mean_dist += sqrt( (points[i][0] - mean_x) * (points[i][0] - mean_x)  +
      (points[i][1] - mean_y) * (points[i][1] - mean_y) );
  }
  mean_dist /= static_cast<double>(num_points);

  // compute tranformation matrix such that mean is zero and mean distance is sqrt2
  K.SetZero();
  K[0][0] = M_SQRT2/mean_dist;
  K[0][2] = -mean_x/mean_dist;
  K[1][1] = M_SQRT2/mean_dist;
  K[1][2] = -mean_y/mean_dist;
  K[2][2] = 1.0;

  // now normalize all the points
  for (size_t i = 0; i < num_points; ++i) 
  {
    K.Mult(points[i], npoints[i]);
    //double z = 1/pt[2];
    //pt.MultIP(z);
  }
}


size_t sfmData::matcher::ransacF(const std::vector<ipc::SIFT_Feature>& feat1, 
  const std::vector<ipc::SIFT_Feature>& feat2, 
  const std::vector<std::pair<int, int> >& matches,      
  std::vector<std::pair<int, int> >& inliers_matches,
  FundMatrixEstimator& F_Estimator,
  MatcherContainer & container) 
{

  //	std::vector<std::pair<int, int> > matches_return;

  size_t num_inliers = 0;
  const size_t num_putatives = matches.size();
  BIAS::Matrix3x3<double> F; F.SetZero();
  BIAS::Matrix3x3<double> solution; solution.SetZero();
  BIAS::Matrix3x3<double> K1; K1.SetZero();
  BIAS::Matrix3x3<double> K2; K2.SetZero();

  // original points
  container.pts1.resize(num_putatives);
  container.pts2.resize(num_putatives);

  // normalized points
  container.npts1.resize(num_putatives);
  container.npts2.resize(num_putatives);

  container.inliers.clear();
  container.inliers.reserve(num_putatives);

  // temporary storage
  BIAS::Vector3<double> pt1(0.0, 0.0, 1.0);
  BIAS::Vector3<double> pt2(0.0, 0.0, 1.0);

  if (num_putatives >= 7)
  {
    for (unsigned int matchIdx = 0; matchIdx < num_putatives; ++matchIdx)
    {
      size_t idx1 = matches[matchIdx].first;
      size_t idx2 = matches[matchIdx].second;

      pt1.Set(feat1[idx1].x, feat1[idx1].y, 1.0);
      pt2.Set(feat2[idx2].x, feat2[idx2].y, 1.0);

      container.pts1[matchIdx] = pt1;
      container.pts2[matchIdx] = pt2;
    }

#if 1
    // normalize data
    normalizePoints(container.pts1, container.npts1, K1);
    normalizePoints(container.pts2, container.npts2, K2);

    // initialize ransac estimator with current data parameters
    F_Estimator.initData(container.npts1, container.npts2);
    F_Estimator.initNormalization(container.pts1, container.pts2, K1,K2);
    // compute solution
    num_inliers = F_Estimator.solveMaster(solution, container.inliers);

    F_Estimator.deallocateData();
#else
    ransac_conf rc;
    ransac_conf_init(4, 7, 3, 2, 9, 9, 256, 0.001f, cuda_F7p_normalize, cuda_F7p_fit, cuda_F_eval, &rc);

    int n_pts = int(container.pts1.size());

    float * pre_data = new float[4*n_pts];
    int j = 0;
    for(int i = 0; i < n_pts; i++)
    {
      if(container.pts1[i][2] > 0 && container.pts2[i][2] > 0)
      {
        pre_data[0*n_pts + j] = container.pts1[i][0] / container.pts1[i][2];
        pre_data[1*n_pts + j] = container.pts1[i][1] / container.pts1[i][2];
        pre_data[2*n_pts + j] = container.pts2[i][0] / container.pts2[i][2];
        pre_data[3*n_pts + j] = container.pts2[i][1] / container.pts2[i][2];
        j++;
      }
      else
      {
      }
    }
    n_pts = j;

    struct ransac_params rp;
    ransac_params_init(n_pts, &rp);
    ransac_params_allocate(rc, &rp);

    float K[18];
    cuda_F7p_normalize2(pre_data, n_pts, rp.data, K);
    delete[] pre_data;

    float F[9];
    cuda_ransac(rc, rp, F);

    for(int i = 0; i < n_pts; i++)
    {
      container.inliers.push_back(rp.h_inlier_indices[i] == 1);
    }

    num_inliers = rp.n_inliers;

    ransac_params_free(rp);
#endif

    inliers_matches.resize(num_inliers);
    if (num_inliers) {
      for(size_t j = 0, pos=0; j < num_putatives; ++j)
        if(container.inliers[j]) 
          inliers_matches[pos++] = matches[j];
    }

  }

  return num_inliers;
}



size_t sfmData::matcher::ransacE(const std::vector<ipc::SIFT_Feature>& feat1, 
  const std::vector<ipc::SIFT_Feature>& feat2,  
  const double * k1,
  const double * k2,
  const std::vector<std::pair<int, int> >& matches,      
  std::vector<std::pair<int, int> >& inliers_matches,
  EssentialMatrixEstimator& E_Estimator,
  MatcherContainer & container,
  BIAS::Matrix3x3<double> & solution) 
{

  //	std::vector<std::pair<int, int> > matches_return;

  size_t num_inliers = 0;
  const size_t num_putatives = matches.size();
  BIAS::Matrix3x3<double> E; E.SetZero();
  solution.SetZero();
  BIAS::Matrix3x3<double> inv_K1; inv_K1.SetZero();
  BIAS::Matrix3x3<double> inv_K2; inv_K2.SetZero();

  // original points
  container.pts1.resize(num_putatives);
  container.pts2.resize(num_putatives);

  // normalized points
  container.npts1.resize(num_putatives);
  container.npts2.resize(num_putatives);

  container.inliers.clear();
  container.inliers.reserve(num_putatives);



  // temporary storage
  BIAS::Vector3<double> pt1(0.0, 0.0, 1.0);
  BIAS::Vector3<double> pt2(0.0, 0.0, 1.0);

  if (num_putatives >= 5)
  {
    for (unsigned int matchIdx = 0; matchIdx < num_putatives; ++matchIdx)
    {
      size_t idx1 = matches[matchIdx].first;
      size_t idx2 = matches[matchIdx].second;

      pt1.Set(feat1[idx1].x, feat1[idx1].y, 1.0);
      pt2.Set(feat2[idx2].x, feat2[idx2].y, 1.0);

      container.pts1[matchIdx] = pt1;
      container.pts2[matchIdx] = pt2;
    }


    // normalize data
    inv_K1[0][0] = 1.0 / k1[0];//k1->f; 
    inv_K1[1][1] = 1.0 / k1[0];//k1->f ; 
    inv_K1[0][2] = - k1[1] / k1[0]; 
    inv_K1[1][2] = -k1[2] / k1[0];
    inv_K1[2][2] = 1.0;

    inv_K2[0][0] = 1.0 / k2[0];//->f; 
    inv_K2[1][1] = 1.0 / k2[0];//->f ; 
    inv_K2[0][2] = - k2[1] / k2[0];; 
    inv_K2[1][2] = -k2[2] /k2[0];;
    inv_K2[2][2] = 1.0;

    for (size_t idx=0; idx < num_putatives; ++idx) {
      container.npts1[idx][0] = (container.pts1[idx][0] - k1[1]) / k1[0];//->f;
      container.npts1[idx][1] = (container.pts1[idx][1] - k1[2]) / k1[0];//->f;
      container.npts1[idx][2] = 1.0;

      container.npts2[idx][0] = (container.pts2[idx][0] - k2[1]) / k2[0];//->f;
      container.npts2[idx][1] = (container.pts2[idx][1] - k2[2]) / k2[0];//->f;
      container.npts2[idx][2] = 1.0;
    }

    // initialize ransac estimator with current data parameters
    E_Estimator.initData(container.npts1, container.npts2);
    E_Estimator.initNormalization(container.pts1, container.pts2, inv_K1,inv_K2);
    // compute solution
    num_inliers = E_Estimator.solveMaster(solution, container.inliers);
    E_Estimator.clear();
    inliers_matches.resize(num_inliers);
    if (num_inliers) {
      for(size_t j = 0, pos=0; j < num_putatives; ++j)
        if(container.inliers[j]) 
          inliers_matches[pos++] = matches[j];
    }

  }

  return num_inliers;
}


#endif

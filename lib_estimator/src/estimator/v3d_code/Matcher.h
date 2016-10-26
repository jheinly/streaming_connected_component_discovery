#ifndef SFMDATA_MATCHER_H
#define SFMDATA_MATCHER_H

#include <vector>
#include <utility>

#ifdef WIN32
//this only compiles under windows
#ifdef _USE_MATH_DEFINES
	#undef _USE_MATH_DEFINES
#endif
#ifdef USE_BIAS
  #include <Base/Math/Matrix.hh>
#endif
#ifndef _USE_MATH_DEFINES
	#define _USE_MATH_DEFINES
#endif
#ifdef USE_BIAS
    #include "FundMatrixEstimator.h"
    #include "EssentialMatrixEstimator.h"
#endif
#endif

//#define V3DLIB_ENABLE_CUDA
//#include <V3D/CUDA/v3d_cudamatching.h>
//#include <V3D/CUDA/BinaryDescriptorMatcher/BinaryDescriptorMatcher.h>

//#include <estimator/v3d_code/cc_filesystem.h>
#include <estimator/v3d_code/Features.h>
//#include <estimator/v3d_code/MatchingTask.h>
#include <estimator/v3d_code/ocv_arrsac_E_5pts.h>

#include <opencv2/opencv.hpp>

//This is a set of function to perfor robust fundamental matrix computation this are static function because it makes the most sense to me. (PFG 5/2010)

namespace sfmData {
  namespace matcher {

    struct MatcherContainer {
#ifdef USE_BIAS
      std::vector< BIAS::Vector3<double> > pts1;
      std::vector< BIAS::Vector3<double> > npts1;

      std::vector< BIAS::Vector3<double> > pts2;
      std::vector< BIAS::Vector3<double> > npts2;
#endif
      void * des1;
      void * des2;

      std::vector<bool> inliers;

      template<typename T>
      T* getDes1()
      {
        return (T*)des1;
      }

      template<typename T>
      T* getDes2()
      {
        return (T*)des2;
      }

      template<typename T>
      void setDes1(T* t)
      {
        des1 = (void*)t;
      }

      template<typename T>
      void setDes2(T* t)
      {
        des2 = (void*)t;
      }
    };

    struct MatcherWrapper
    {
      //V3D_CUDA::SIFT_Matching sift;
      //V3D_CUDA::BinaryDescriptorMatcher harrisBrief;
    };

    /*size_t matchImages(const V3D::Features& feat1,
		       const V3D::Features& feat2,
		       std::vector< std::pair<int, int> >& matches,
		       size_t maxNumFeatures,
           MatcherWrapper & matcherWrapper,
		       MatcherContainer & container);*/

    /*size_t computeNormalizedFloatingDesc(float*& floatDesc,
                                         const V3D::Features& features,
                                         size_t maxNumSIFT);*/


    /*size_t ransacF_ocv(const V3D::Features& feat1,
      const V3D::Features& feat2,
      const std::vector<std::pair<int, int> >& matches,
      std::vector<std::pair<int, int> >& inliers_matches,
      ocv_lib::ARRSAC_fundamental& F_Estimator,
      MatcherContainer & container,
      cv::Mat & solution) ;*/

    /*size_t ransacH_ocv(const V3D::Features& feat1,
      const V3D::Features& feat2,
      const std::vector<std::pair<int, int> >& matches,
      std::vector<std::pair<int, int> >& inliers_matches,
      ocv_lib::ARRSAC_homography& H_Estimator,
      MatcherContainer & container);

    size_t ransacH_ocv(const V3D::Features& feat1,
      const V3D::Features& feat2,
      const std::vector<std::pair<int, int> >& matches,
      std::vector<std::pair<int, int> >& inliers_matches,
      ocv_lib::ARRSAC_homography& H_Estimator,
      MatcherContainer & container,
	  cv::Mat & solution);*/

    size_t ransacE_ocv(const V3D::Features& feat1,
		       const V3D::Features& feat2,
		       const double * k1,
		       const double * k2,
		       const std::vector<std::pair<int, int> >& matches,
		       std::vector<std::pair<int, int> >& inliers_matches,
		       ocv_lib::ARRSAC_essential& E_Estimator_ocv,
		       MatcherContainer & container,
		       cv::Mat & solution,
           bool shouldCheckForLineDegeneracy);

    bool checkForLineDegeneracy(const V3D::Features& feat1,
		                            const V3D::Features& feat2,
                                std::vector<std::pair<int, int> >& inlier_matches,
                                float inlier_threshold1,
                                float inlier_threshold2);


#ifdef USE_BIAS
    //this is only defines for windows as we did not compile BIAS under mac
    //the code itself is not needed

    size_t ransacE(
		   const std::vector<ipc::SIFT_Feature>& feat1,
		   const std::vector<ipc::SIFT_Feature>& feat2,
		   const double * k1,
		   const double * k2,
		   const std::vector<std::pair<int, int> >& matches,
		   std::vector<std::pair<int, int> >& inliers_matches,
		   EssentialMatrixEstimator& E_Estimator,
		   MatcherContainer & container,
		   BIAS::Matrix3x3<double> & solution);

    size_t ransacF(const std::vector<ipc::SIFT_Feature>& feat1,
		   const std::vector<ipc::SIFT_Feature>& feat2,
		   const std::vector<std::pair<int, int> >& matches,
		   std::vector<std::pair<int, int> >& inliers_matches,
		   FundMatrixEstimator& F_Estimator,
		   MatcherContainer & container);

    void normalizePoints(const std::vector< BIAS::Vector3<double> >& points,
			 std::vector< BIAS::Vector3<double> >& npoints,
			 BIAS::Matrix3x3<double>& K);

#endif

  }
}

#endif

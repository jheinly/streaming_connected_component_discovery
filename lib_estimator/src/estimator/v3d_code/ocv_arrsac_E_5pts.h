#ifndef OCV_ARRSAC_E_5PTS_HH
#define OCV_ARRSAC_E_5PTS_HH

#include <vector>


#include <estimator/v3d_code/ocv_arrsac.h>
#include <estimator/v3d_code/OCV_utils.h>
#include <estimator/v3d_code/essentialMat.h>
#include <estimator/v3d_code/v3d_ocv.h>
#include <estimator/v3d_code/v3d_poseutilities.h> //5 point algorithm.

namespace ocv_lib {
  class ARRSAC_essential: public ARRSAC
  {
	  public:
		  ARRSAC_essential()
		  {
		  }

		  ~ARRSAC_essential()
		  {
		  }

		   void initData(const cv::Mat & npts1, const cv::Mat & npts2);

		    void deallocateData();

		    int getSampleSolutions(const std::vector<size_t>& sample, std::vector< cv::Mat >& solutions);
		    bool evaluateSolution(const cv::Mat& solution, double An, int &numInliers, size_t &numPointsTested, int evalMethod = 0) const;
        int findInliers(cv::Mat & solution, std::vector<bool>& inliers) const;

		    void clear();

		    void initNormalization(const cv::Mat & pts1, const cv::Mat & pts2,
                                const cv::Mat & K1, const cv::Mat & K2,
                                const cv::Mat &dist1,const cv::Mat &dist2);

        double getInlierThresholdInPixel() const
        { return m_inlierThresholdInPixel; }

	  private:

      double computeSampsonError(const cv::Mat & model , unsigned int i) const;
		  double computePointToLine(const cv::Mat & model,unsigned int i) const;

		  // input data
		  double m_inlierThresholdInPixel;

      std::vector<V3D::Matrix3x3d> m_Es;

      cv::Mat m_npts1;
      cv::Mat m_npts2;

      cv::Mat m_pts1;
      cv::Mat m_pts2;

      cv::Mat m_K1;
      cv::Mat m_K2;


      double m_x1[5];
		  double m_y1[5];
		  double m_x2[5];
		  double m_y2[5];
		  double m_ff[30][3];

		  //const unsigned char m_num_sample = 5;

  };
}

#endif


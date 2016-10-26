#ifndef OCV_ARRSAC_HH
#define OCV_ARRSAC_HH
#define NOMINMAX

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>


#include <opencv2/opencv.hpp>

namespace ocv_lib {
  class ARRSAC
  {
  public:
    ARRSAC() {};
    virtual ~ARRSAC() {} ;
    void init(size_t minSampleSize, double inlierThreshold,
	      size_t maxHypotheses, size_t maxSolutionsPerSample,
	      size_t blockSize);

    void setInlierThreshold(double inlierThreshold);
    size_t solveMaster(cv::Mat & solution, std::vector<bool>& inliers);

    // functions for derived class
    virtual void initData(const cv::Mat & nPts1, const cv::Mat & nPts2) = 0;
    virtual int getSampleSolutions(const std::vector<size_t>& samples,
				   std::vector<cv::Mat>& solutions) = 0;
    virtual bool evaluateSolution(const cv::Mat& solution, double An,
				  int &ninliers, size_t &numPointsTested,
				  int evalMethod = 0) const = 0;
    virtual int findInliers(cv::Mat& solution,
			    std::vector<bool>& inliers) const = 0;

    void ARRSACsetInlierThreshold(double inlierThreshold)
    {
      m_inlierThreshold = inlierThreshold;
    }

	protected:
    std::vector<unsigned int> m_sample;		// holds the sample indices

    // common parameters
    size_t m_minSampleSize;
    double m_inlierThreshold;
    size_t m_maxHypotheses;
    size_t m_maxSolutionsPerSample;
    int m_numDataPoints;
    size_t m_blockSize;
    double m_confThreshold;

    // SPRT parameters
    double m_SPRT_tM;
    double m_SPRT_mS;
    double m_SPRT_delta;
    double m_SPRT_epsilon;

  private:
    inline double designSPRTTest();
    inline bool generateRandomSample(std::vector<size_t>& sample);
};

}
#endif //OCV_ARSSAC_HH

#include <estimator/v3d_code/ocv_arrsac_E_5pts.h>


using namespace ocv_lib;
void ARRSAC_essential::initData(const cv::Mat & npts1, const cv::Mat & npts2)
{
  m_numDataPoints = npts1.cols; 
  if (m_numDataPoints != npts1.cols) {
    std::cerr << " Number of target 2d points different than the number of source 2d points ! ARRSAC_homography!" << std::endl;
    exit(-1);
  }

  m_npts1 = npts1;
  m_npts2 = npts2;
}
void ARRSAC_essential::clear() {
	m_inlierThreshold = m_inlierThresholdInPixel;
}

void ARRSAC_essential::initNormalization(const cv::Mat & pts1, const cv::Mat & pts2,
                              const cv::Mat & K1, const cv::Mat & K2, 
                              const cv::Mat &dist1,const cv::Mat &dist2)
{
  m_numDataPoints = pts1.cols; 
  m_inlierThresholdInPixel = m_inlierThreshold;
    
  m_inlierThreshold = m_inlierThreshold / ( std::max( 0.5 * ( K1.at<double>(0,0) + K1.at<double>(1,1) ),
                                                      0.5 * ( K2.at<double>(0,0) + K2.at<double>(1,1) )) );

  m_pts1 = pts1;
  m_pts2 = pts2;

  m_K1 = K1;
  m_K2 = K2;
  
  m_npts1.create(2,pts1.cols, CV_64FC1);
  m_npts2.create(2,pts1.cols, CV_64FC1);
  ocv_lib::undistortPoints(pts1, K1, dist1, m_npts1);
  ocv_lib::undistortPoints(pts2, K2, dist2, m_npts2);
}


int ARRSAC_essential::getSampleSolutions(const std::vector<size_t>& sample, std::vector< cv::Mat >& solutions)
{
  solutions.clear();
  m_Es.clear();



  //computeEs
  for (unsigned char idx = 0; idx < m_minSampleSize; ++idx) {
	  m_x1[idx] = m_npts1.at<double>(0, static_cast<int>(sample[idx]));
	  m_y1[idx] = m_npts1.at<double>(1, static_cast<int>(sample[idx]));
    m_x2[idx] = m_npts2.at<double>(0, static_cast<int>(sample[idx]));
    m_y2[idx] = m_npts2.at<double>(1, static_cast<int>(sample[idx]));
  }
  
  int indentic_points =0;
  for( unsigned char idx = 0; idx < m_minSampleSize; ++idx) {
    if ((m_x1[idx]==m_x2[idx])&&(m_y1[idx]==m_y2[idx])) 
      ++indentic_points;
  }
  if (indentic_points >2 ) {
    //std::cerr<< "problem ARRSAC_essential indentical inputs?" << endl;
    return 0;
  }

  for (int i = 0; i < m_minSampleSize - 1; ++i)
  {
    for (int j = i + 1; j < m_minSampleSize; ++j)
    {
      if ((m_x1[i] == m_x1[j]) && (m_y1[i] == m_y1[j]))
      {
        return 0;
      }

      if ((m_x2[i] == m_x2[j]) && (m_y2[i] == m_y2[j]))
      {
        return 0;
      }
    }
  }

  V3D::computeEssentialsFromFiveCorrs(m_x1, m_y1, m_x2, m_y2, m_Es);

  for(size_t idx=0; idx < m_Es.size(); ++idx) {
     cv::Mat E(3,3,CV_64FC1);
     for (int i=0; i<3;++i)
       for (int j=0; j<3;++j) 
            E.at<double>(i,j) = m_Es[idx][i][j];
    
      solutions.push_back(E);
  }

  return static_cast<int> (m_Es.size());
}
double ARRSAC_essential::computeSampsonError(const cv::Mat & model , unsigned int i) const
	{
		double x2tFx1 , Fx1[3], Ftx2[3];

		for ( int j = 0 ; j < 3 ; j ++ ) 
		{
			Fx1[j] = model.at<double>(j,0)* m_npts1.at<double>(0,i) 
                +  model.at<double>(j,1)* m_npts1.at<double>(1,i) 
                +  model.at<double>(j,2);
			Ftx2[j] = model.at<double>(0,j)* m_npts2.at<double>(0,i) 
                +  model.at<double>(1,j)* m_npts2.at<double>(1,i) 
                +  model.at<double>(2,j);
        
		}
		x2tFx1 =  m_npts2.at<double>(0,i)   * Fx1[0]+ m_npts2.at<double>(1,i) * Fx1[1]+ Fx1[2];
		return (x2tFx1* x2tFx1) / (Fx1[0]*Fx1[0] + Fx1[1]*Fx1[1] + Ftx2[0]*Ftx2[0] + Ftx2[1]*Ftx2[1]);		
	}

double ARRSAC_essential::computePointToLine(const cv::Mat & model , unsigned int i) const
{
  double x2tFx1 , Fx1[3], Ftx2[3];

		for ( int j = 0 ; j < 3 ; j ++ ) 
		{
			Fx1[j] = model.at<double>(j,0)* m_npts1.at<double>(0,i) 
                +  model.at<double>(j,1)* m_npts1.at<double>(1,i) 
                +  model.at<double>(j,2);
			Ftx2[j] = model.at<double>(0,j)* m_npts2.at<double>(0,i) 
                +  model.at<double>(1,j)* m_npts2.at<double>(1,i) 
                +  model.at<double>(2,j);
        
		}
		x2tFx1 =  m_npts2.at<double>(0,i)   * Fx1[0]+ m_npts2.at<double>(1,i) * Fx1[1]+ Fx1[2];
    return  0.5 * abs(x2tFx1) * (1.0 / sqrt(Fx1[0]*Fx1[0] + Fx1[1]*Fx1[1])  + 1.0 / sqrt( Ftx2[0]*Ftx2[0] + Ftx2[1]*Ftx2[1]) );
   
}

bool ARRSAC_essential::evaluateSolution(const cv::Mat& solution, double An, int &numInliers, size_t &numPointsTested, int evalMethod) const
{
	bool good_flag = true;
	double lambdaj, lambdaj_1 = 1.0;
	numInliers = 0;
	numPointsTested = 0;

	
	// evaluate error for data points
	for (int i = 0; i < m_numDataPoints; ++i)
	{
	  double err = computePointToLine(solution, i);
	  //double err = computeSampsonError( solution, i);
		
	  if (err < m_inlierThreshold) 
	    ++numInliers;
  		

		if (evalMethod == 1)
		{
			if (err < m_inlierThreshold)
			{			
	      lambdaj = lambdaj_1 * (m_SPRT_delta/m_SPRT_epsilon);
	    } else {
	      lambdaj = lambdaj_1 * ( (1 - m_SPRT_delta)/(1 - m_SPRT_epsilon) );
	    }

	    if (lambdaj > An) {
	      good_flag = false;
	      numPointsTested = i+1;
	      return good_flag;
	    } else {
	      lambdaj_1 = lambdaj;
	    }
    }
  }
	
  numPointsTested = m_numDataPoints;
	return good_flag;
}

int ARRSAC_essential::findInliers(cv::Mat & solution, std::vector<bool>& inliers) const
{
  int ninliers = 0;
  inliers.resize(m_numDataPoints);
  
  // evaluate error for data points
  for (int i = 0; i < m_numDataPoints; ++i)
  {
    double err = computePointToLine(solution, i);
	  //double err = computeSampsonError( solution, i);

    if (inliers[i] =(err < m_inlierThreshold))
    {
      ++ninliers;
    }
  }

  if (ninliers > 5)
  {
    V3D::Matrix3x3d Kleft; 
    V3D::Matrix3x3d Kright; 

    ocv_lib::translate(m_K1, Kleft);
    ocv_lib::translate(m_K2, Kright);
  
    V3D::Matrix3x3d R_v3d;
    V3D::Vector3d T_v3d;

    std::vector<V3D::Vector2d> left(ninliers);
    std::vector<V3D::Vector2d>  right(ninliers);

    for (int i=0, inlier_idx=0; i < m_numDataPoints; ++i)
    {
      if (inliers[i])
      {
        left[inlier_idx][0] = m_pts1.at<double>(0,i);
        left[inlier_idx][1] = m_pts1.at<double>(1,i);
        right[inlier_idx][0] = m_pts2.at<double>(0,i);
        right[inlier_idx][1] = m_pts2.at<double>(1,i);
        ++inlier_idx;
      }
    }

    cv::Mat R1(3,3,CV_64FC1);
    cv::Mat R2(3,3,CV_64FC1);
    cv::Mat t1(3,1,CV_64FC1);
    cv::Mat t2(3,1,CV_64FC1);

    decomposeEssentialMat(solution, R1, R2, t1, t2);
    
    ocv_lib::translate(R1,R_v3d);
    ocv_lib::translate(t1,T_v3d);


    
    V3D::refineRelativePose(left, right, Kleft, Kright, R_v3d, T_v3d, m_inlierThresholdInPixel);

    ocv_lib::translate(R_v3d, R1);
    ocv_lib::translate(T_v3d, t1);

    computeEssentialFromRT(R1,t1,solution);

    int ninliers_after_nl = 0;

    // evaluate error for data points
    for (int i = 0; i < m_numDataPoints; ++i)
    {
      double err = computePointToLine(solution, i);
	    //double err = computeSampsonError( solution, i);

      if (inliers[i] =(err < m_inlierThreshold))
      {
        ++ninliers_after_nl;
      }
    }
    return ninliers_after_nl;
  }
  return ninliers;
}

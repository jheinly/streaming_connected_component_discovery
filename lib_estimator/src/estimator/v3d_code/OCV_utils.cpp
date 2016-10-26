#include <estimator/v3d_code/OCV_utils.h>

void saveMatForMatlab(const cv::Mat & mat, const std::string & fn) {
  using std::endl;
  std::ofstream ofs((fn+".m").c_str());

  ofs << fn << " = [ ..." << endl;

  for (int row=0; row < mat.rows; ++row) {
    for (int col=0; col < mat.cols; ++col) 
      ofs << mat.at<double>(row,col) << " ";
    if (row != mat.rows-1)
        ofs << ";";
    ofs << "..." << endl;
  }
  ofs << "];"<< endl;
        
    ofs.close();
}

  void ocv_lib::getReprojectionError(const cv::Mat & pts_3d, const cv::Mat & obs, const cv::Mat & camera_matrix, const cv::Mat & dist, const cv::Mat & R, const cv::Mat & t, double * err_)
  {
    int nPoints = pts_3d.cols;
    //err.resize(nPoints);
    cv::Mat meas;
    //cv::Mat cv_err(1,nPoints, CV_64FC1, err_);
    //cout << "projecting" << endl;
    projectPoints(pts_3d, camera_matrix, dist, R, t, meas);

    meas = meas - obs;
    //cout << "measuring" << endl;
    for (int i=0; i < nPoints; ++i) 
    {
      err_[i] = sqrt(meas.at<double>(0,i)*meas.at<double>(0,i) + meas.at<double>(1,i)*meas.at<double>(1,i));
    }
  }

void ocv_lib::projectPoints(const cv::Mat & pts_3d, const cv::Mat & camera_matrix, const cv::Mat & dist, const cv::Mat & R, const cv::Mat & t,
    cv::Mat & pts_2d)
  {
    
    int nPoints = pts_3d.cols;
    bool return_3dim_pts;
    if (nPoints != pts_2d.cols) {
      pts_2d.create(2,nPoints,CV_64FC1);
      return_3dim_pts = false;
    } else {
      return_3dim_pts = (pts_2d.rows == 3);
    }
    if (camera_matrix.at<double>(0,1) != 0) 
      std::cerr << "Problem in projectPoints camera_matrix without 0 skew!!!" << std::endl;

    cv::Mat m = R * pts_3d;// + t; //todo this might be a bottle neck?
    m.row(0) += t.at<double>(0);
    m.row(1) += t.at<double>(1);
    m.row(2) += t.at<double>(2);
    const double *k = dist.ptr<double>();
    
    //get camera info
    double fx = camera_matrix.at<double>(0,0);
    double fy = camera_matrix.at<double>(1,1);
    double cx = camera_matrix.at<double>(0,2);
    double cy = camera_matrix.at<double>(1,2);
    
    for (int ptsIdx=0; ptsIdx < nPoints; ++ptsIdx) {
      double r2, r4, a1, a2, a3, cdist;
      double xd, yd;
      double z = m.at<double>(2,ptsIdx); 
      z = z ? 1./z : 1; //check point at infinity.
      double x = z * m.at<double>(0,ptsIdx);; 
      double y = z * m.at<double>(1,ptsIdx);;

      r2 = x*x + y*y;
      r4 = r2*r2;
      a1 = 2*x*y;
      a2 = r2 + 2*x*x;
      a3 = r2 + 2*y*y;
      cdist = 1 + k[0]*r2 + k[1]*r4; 
      
      xd = x*cdist + k[2]*a1 + k[3]*a2;
      yd = y*cdist + k[2]*a3 + k[3]*a1;

      pts_2d.at<double>(0,ptsIdx) = xd*fx + cx;
      pts_2d.at<double>(1,ptsIdx) = yd*fy + cy;

      if (return_3dim_pts) 
        pts_2d.at<double>(2,ptsIdx) = 1.0;
    }
}

void ocv_lib::undistortPoints(const cv::Mat & pts_dist, const cv::Mat & camera_matrix, const cv::Mat & dist, cv::Mat & pts_undist) 
{
  int nPoints = pts_dist.cols;
  bool return_3dim_pts = (pts_undist.rows == 3);

 if (camera_matrix.at<double>(0,1) != 0) 
   std::cerr << "Problem in projectPoints camera_matrix without 0 skew!!!" << std::endl;

  //get camera info
  double x,y,x0,y0; 
  double ifx = 1.0 / camera_matrix.at<double>(0,0);
  double ify = 1.0 / camera_matrix.at<double>(1,1);
  double cx = camera_matrix.at<double>(0,2);
  double cy = camera_matrix.at<double>(1,2);
  const double *k = dist.ptr<double>();
  int nIter = 5;
   
  
   for (int ptsIdx=0; ptsIdx < nPoints; ++ptsIdx) 
   {
      x = pts_dist.at<double>(0,ptsIdx);
      y = pts_dist.at<double>(1,ptsIdx);
      x0 = x = (x - cx)*ifx;
      y0 = y = (y - cy)*ify;



      // compensate distortion iteratively
      for( int j = 0; j < nIter; j++ )
      {
        double x2 = x*x;
        double y2 = y*y;
        double xy = x*y;
        double r2 = x2 + y2;
        double icdist = (1 )/(1 + ((k[1])*r2 + k[0])*r2);
        double deltaX = 2*k[2]*xy + k[3]*(r2 + 2*x2);
        double deltaY = k[2]*(r2 + 2*y2) + 2*k[3]*xy;
        x = (x0 - deltaX)*icdist;
        y = (y0 - deltaY)*icdist;
      }

      pts_undist.at<double>(0,ptsIdx) = x;
      pts_undist.at<double>(1,ptsIdx) = y;
      if (return_3dim_pts) 
        pts_undist.at<double>(2,ptsIdx) = 1.0;

   }
}

double getSquaredRerojectionError(const cv::Mat & pt_3d, const cv::Mat & pt_2d, 
                                  const cv::Mat & K, const cv::Mat & dist, const cv::Mat & R, const cv::Mat & t)
{
  double p2d_raw[2];
  cv::Mat p2d_(2,1,CV_64FC1,p2d_raw);

  ocv_lib::projectPoints(pt_3d,K,dist,R,t,p2d_);  //K * (R * pt_3d + t);
  
  double err_x = pt_2d.at<double>(0) - p2d_raw[0];
  double err_y = pt_2d.at<double>(1) - p2d_raw[1];
  return err_x*err_x + err_y*err_y;
}

double getMeanSquaredRerojectionError(const std::vector<cv::Mat> & ps, 
                                      const std::vector<cv::Mat> & Ks, 
                                      const std::vector<cv::Mat> & dists, 
                                      const std::vector<cv::Mat> & Rs, 
                                      const std::vector<cv::Mat> & ts, 
                                      const cv::Mat & M) 
{
  size_t nViews = ps.size();
  double err = 0.0;
  for (size_t viewIdx=0; viewIdx < nViews; ++viewIdx)
  {
    err += sqrt(getSquaredRerojectionError(M, ps[viewIdx],Ks[viewIdx], dists[viewIdx], Rs[viewIdx], ts[viewIdx]));
  }
  
  return err / nViews;
}








size_t checkRerojectionError(const cv::Mat & pts_3d, const cv::Mat & pts_2d, 
                           const cv::Mat & K, const cv::Mat & dist, const cv::Mat & R, const cv::Mat & t, 
                           std::vector<bool> & status, 
                           double threshold)
{
  threshold = threshold * threshold; // to avoid doing any sqrt.
  size_t nInliers = 0;
  int nPoints = pts_3d.cols;
  for (int idx=0; idx < nPoints; ++idx) 
    if (status[idx]) {
      double err = getSquaredRerojectionError(pts_3d.col(idx), pts_2d.col(idx), K, dist,R, t);
      if (status[idx] = ( err < threshold)) 
        ++nInliers;

    }
  return nInliers;
}
cv::Mat PfromRt(const cv::Mat & R, const cv::Mat & t) {
  using namespace cv;
  
  Mat P(3,4, CV_64FC1);

  for (int i=0; i<3; ++i) {
    Mat P_col = P.col(i); 
    R.col(i).copyTo(P_col);  
  }
  
  Mat P_t = P.col(3); 
  t.copyTo(P_t);

  return P;
}

void ocv_lib::getCameraLocation(const cv::Mat& R, const cv::Mat& t, cv::Mat& C) {
  	using namespace cv;
  C = -R.t() * t;
}
void ocv_lib::getCameraLocation(const cv::Mat& P, cv::Mat& C) {
  getCameraLocation(P.colRange(0,3), P.col(3), C);
}

bool checkForForwardMotion(const cv::Mat& R, const cv::Mat& t) {
  double C_[3];
  cv::Mat C(3,1,CV_64FC1,C_);
  ocv_lib::getCameraLocation(R,t, C);

  return (abs(C_[2]) < 0.95);
}

void computeCofactorsMat(const cv::Mat & M, 
						 cv::Mat &  cofact)
{
//Note it does not return a transpose of the cofactors so
// be careful when using it.

	using namespace cv;
	Mat M1 = M.col(0);
	Mat M2 = M.col(1);
	Mat M3 = M.col(2);

	Mat cofact1 = cofact.col(0);
	Mat cofact2 = cofact.col(1);
	Mat cofact3 = cofact.col(2);

  M2.cross(M3).copyTo(cofact1);
	M3.cross(M1).copyTo(cofact2);
	M1.cross(M2).copyTo(cofact3);
 }

void projectOnSO3(const cv::Mat& src, cv::Mat& dest) {

	using namespace cv;
	Mat m(3,1,CV_64FC1);

	Rodrigues(src,m);
	Rodrigues(m,dest);
}

void ppv(const cv::Mat& t, cv::Mat & res) {

//This is done in case of a getCol access.
	res.at<double>(0,0) = 0;
	res.at<double>(0,1) = -1.0 * t.at<double>(2,0);
	res.at<double>(0,2) = t.at<double>(1,0);
	res.at<double>(1,0) = t.at<double>(2,0);
    res.at<double>(1,1) = 0.0;
	res.at<double>(1,2) = -1.0 * t.at<double>(0,0);
	res.at<double>(2,0) = -1.0 * t.at<double>(1,0);
	res.at<double>(2,1) = t.at<double>(0,0);
	res.at<double>(2,2) = 0.0;
}
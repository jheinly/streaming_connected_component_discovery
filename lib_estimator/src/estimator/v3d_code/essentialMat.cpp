#include <estimator/v3d_code/essentialMat.h>

#include <iostream>

void computeEssentialFromRT(const cv::Mat & R,
                            const cv::Mat & t,
                            cv::Mat E) 
{
  double ppv_t_[9];
  cv::Mat ppv_t(3,3,CV_64FC1,ppv_t_);
  ppv(t, ppv_t);

  E = ppv_t * R;
}

void decomposeEssentialMat(const cv::Mat & E, 
						   cv::Mat & R1, 
						   cv::Mat & R2, 
						   cv::Mat & t1, 
						   cv::Mat & t2) {
using namespace cv;
	Mat cofact(3,3,CV_64FC1);
    Mat biggestCofact(3,1,CV_64FC1);
	Mat ppvt(3,3, CV_64FC1); //clear
    Mat ppvtE(3,3, CV_64FC1); //clear


	Mat EEt = E * E.t(); 
	Scalar traceEEtScalar = trace(EEt);      
	
	Mat cofact1 = cofact.col(0);
    Mat cofact2 = cofact.col(1);
    Mat cofact3 = cofact.col(2);
    
	computeCofactorsMat(E, cofact);
    
	double normCofact1 = norm(cofact1);
    double normCofact2 = norm(cofact2);
	double normCofact3 = norm(cofact3);

   if ( normCofact1 >  normCofact2 )
	if ( normCofact1 > normCofact3 )
		biggestCofact = cofact1 / normCofact1;
    else
		biggestCofact = cofact3 / normCofact3;
   else
	if ( normCofact2 > normCofact3 )
		biggestCofact = cofact2 / normCofact2;
	else
		biggestCofact = cofact3 / normCofact3;

	t1 = biggestCofact * sqrt( 0.5 * traceEEtScalar.val[0]);
    ppv(t1, ppvt);
	ppvtE = ppvt * E;
    
    double normt = norm(t1);
    double normtSquared = normt * normt;

  
	R1 = (cofact - ppvtE) / normtSquared;
    projectOnSO3(R1, R1);
  
	R2 = (cofact + ppvtE) / normtSquared;
	projectOnSO3(R2, R2);
  

	t1 = t1 / normt;
	t2 = t1 * -1.0;

}




size_t selectCorrectRtUsingCherality(const cv::Mat & R1,
				     const cv::Mat & R2,
				     const cv::Mat & t1,
				     const cv::Mat & t2,
				     const cv::Mat & m1, 
				     const cv::Mat & m2, 
				     std::vector<bool> & status,
				     cv::Mat & R,
				     cv::Mat & t) {
  //todo when cheralityCheck fails run the next one on status || !statusI
  //count the number of input points
  size_t inputNumPoint = 0;
  for (size_t idx=0; idx < status.size(); ++idx) 
    if (status[idx])
      ++inputNumPoint;
  
  std::vector<bool> statusI(status);
	
  size_t NumInfrontPoint = cheralityCheck(R1, t1, m1, m2, statusI);
	
  if ( static_cast<double>(NumInfrontPoint) / static_cast<double>(inputNumPoint) < 0.5 ){
    //less than 50% of the point are in front of R1 t1
    statusI = status;
    NumInfrontPoint = cheralityCheck(R1, t2, m1, m2, statusI);
    if ( static_cast<double>(NumInfrontPoint) / static_cast<double>(inputNumPoint) < 0.5) {
      //less than 50% of the point are in front of R1 t2
      statusI = status;
      NumInfrontPoint = cheralityCheck(R2, t1, m1, m2, statusI);
      if ( static_cast<double>(NumInfrontPoint) / static_cast<double>(inputNumPoint) < 0.5) {
	statusI = status;
	NumInfrontPoint = cheralityCheck(R2, t2, m1, m2, statusI);
	if ( static_cast<double>(NumInfrontPoint) / static_cast<double>(inputNumPoint) < 0.5) {
	  //This should not happens except really bad configuration
    status = vector<bool>(status.size(), false);
	  return 0;					
	} else {
	  R2.copyTo(R);
	  t2.copyTo(t);
	}
      } else {
	R2.copyTo(R); 
	t1.copyTo(t);
      }
    } else {
      R1.copyTo(R); 
      t2.copyTo(t);
    }
  } else {
    R1.copyTo(R); 
    t1.copyTo(t);
  }
								
  status = statusI;
  return NumInfrontPoint;
}

size_t refineRTfromEssentialUsingPointTriangulation( const cv::Mat & pts_src, const cv::Mat & pts_tgt,
                                                    const cv::Mat & K_src, const cv::Mat & K_tgt,
                                                     cv::Mat & E,
												                             cv::Mat & R, cv::Mat & t,
												                              std::vector<bool> & status) 
{
  using namespace cv;
  double R1_[9], R2_[9], t1_[3], t2_[3], d_[4], M_[3];
  d_[0] = d_[1] = d_[2] = d_[3] = 0.0;
  
  Mat M(3,1, CV_64FC1, M_);
  Mat dist(4,1, CV_64FC1, d_);
  Mat m_src(3, pts_src.cols,CV_64FC1);
  Mat m_tgt(3, pts_tgt.cols,CV_64FC1);

  ocv_lib::undistortPoints(pts_src,K_src, dist, m_src);
  ocv_lib::undistortPoints(pts_tgt,K_tgt, dist, m_tgt);
  
  cv::Mat R1(3,3,CV_64FC1,R1_);//Cleared
	cv::Mat R2(3,3,CV_64FC1,R2_);//Cleared
	cv::Mat t1(3,1,CV_64FC1,t1_);//Cleared
	cv::Mat t2(3,1,CV_64FC1,t2_);//Cleared
	
	size_t pointNum = status.size();

  decomposeEssentialMat(E, R1, R2, t1, t2);

  size_t score[4];
  score[0] = score[1] = score[2] = score[3] = 0;
  vector<bool> status_11(pointNum, false);
  vector<bool> status_12(pointNum, false);
  vector<bool> status_21(pointNum, false);
  vector<bool> status_22(pointNum, false);

  cv::Mat P_11 = PfromRt(R1,t1);
  cv::Mat P_12 = PfromRt(R1,t2);
  cv::Mat P_21 = PfromRt(R2,t1);
  cv::Mat P_22 = PfromRt(R2,t2);

  for (size_t idx=0; idx < pointNum; ++idx) {
    
    reconstruct3dPoint( m_src.col( static_cast<int>(idx)), m_tgt.col( static_cast<int>(idx)), P_11, M);
    
    if ( (status_11[idx] = ((M.at<double>(2) > 0.0)&&(checkCherality(R1,t1,M)))) == true )
      ++score[0];

    reconstruct3dPoint( m_src.col( static_cast<int>(idx)), m_tgt.col( static_cast<int>(idx)), P_12, M);
    if ( (status_12[idx] = ((M.at<double>(2) > 0.0)&&(checkCherality(R1,t2,M)))) == true )
      ++score[1];
    
    reconstruct3dPoint( m_src.col( static_cast<int>(idx)), m_tgt.col( static_cast<int>(idx)), P_21, M);
    if ( (status_21[idx] = ((M.at<double>(2) > 0.0)&&(checkCherality(R2,t1,M)))) == true )
      ++score[2];

    reconstruct3dPoint( m_src.col( static_cast<int>(idx)), m_tgt.col( static_cast<int>(idx)), P_22, M);
    if ( (status_22[idx] = ((M.at<double>(2) > 0.0)&&(checkCherality(R2,t2,M)))) == true )
      ++score[3];

    //if ( (status_11[idx]&&status_12[idx])
    //      || (status_11[idx]&&status_21[idx])
    //      || (status_11[idx]&&status_22[idx])
    //      || (status_12[idx]&&status_21[idx])
    //      || (status_12[idx]&&status_22[idx])
    //      || (status_21[idx]&&status_22[idx]))
    //      std::cerr << "How is this possible? I thought it could not be?" << endl;
   }

  size_t currentMax = score[0];
  int bestIndex = 0;

  for (int i=1; i < 4; ++i) {
    if (score[i] > currentMax) {
      currentMax = score[i];
      bestIndex = i;
    }
  }


  switch(bestIndex) {
    case 0:
      status = status_11;
      R1.copyTo(R);
      t1.copyTo(t);
      return score[0];
    case 1:
      status = status_12;
      R1.copyTo(R);
      t2.copyTo(t);
      return score[1];
    case 2:
      status = status_21;
      R2.copyTo(R);
      t1.copyTo(t);
      return score[2];
    case 3:
      status = status_22;
      R2.copyTo(R);
      t2.copyTo(t);
      return score[3];
    default:
      return 0;
  }

   
}

size_t refineRTfromEssentialusingDepthConstraint( const cv::Mat & pts_src, const cv::Mat & pts_tgt,
                                                 const cv::Mat & K_src, const cv::Mat & K_tgt,
                                                 cv::Mat & E,
												                         cv::Mat & R, cv::Mat & t,
												                         std::vector<bool> & status) {
	using namespace cv;
  double R1_[9], R2_[9], t1_[3], t2_[3], d_[4];
  d_[0] = d_[1] = d_[2] = d_[3] = 0.0;

  Mat dist(4,1, CV_64FC1, d_);
  Mat m_src(3, pts_src.cols,CV_64FC1);
  Mat m_tgt(3, pts_tgt.cols,CV_64FC1);

  ocv_lib::undistortPoints(pts_src,K_src, dist, m_src);
  ocv_lib::undistortPoints(pts_tgt,K_tgt, dist, m_tgt);
  
  V3D::Matrix3x3d Kleft; 
  V3D::Matrix3x3d Kright; 

  ocv_lib::translate(K_src, Kleft);
  ocv_lib::translate(K_tgt, Kright);
  
  V3D::Matrix3x3d R_v3d;
  V3D::Vector3d T_v3d;

  cv::Mat R1(3,3,CV_64FC1,R1_);//Cleared
	cv::Mat R2(3,3,CV_64FC1,R2_);//Cleared
	cv::Mat t1(3,1,CV_64FC1,t1_);//Cleared
	cv::Mat t2(3,1,CV_64FC1,t2_);//Cleared
	
	size_t pointNum = status.size();


    size_t inFrontNum = 0;
	size_t consideredNum = pointNum;
  std::vector<V3D::Vector2d> left(consideredNum);
  std::vector<V3D::Vector2d>  right(consideredNum);
    
	//int iterationNum = 0;//iterationNum++;
  decomposeEssentialMat(E, R1, R2, t1, t2);


	while (( consideredNum >= 5) && (consideredNum != inFrontNum)) 
  {
			
    left.resize(consideredNum);
    right.resize(consideredNum);

    for (int i=0, inlier_idx=0; i < pts_src.cols; ++i) {
      if (status[i]) {
        left[inlier_idx][0] = pts_src.at<double>(0,i);
        left[inlier_idx][1] = pts_src.at<double>(1,i);
        right[inlier_idx][0] = pts_tgt.at<double>(0,i);
        right[inlier_idx][1] = pts_tgt.at<double>(1,i);
        ++inlier_idx;
      }
    }

    ocv_lib::translate(R1,R_v3d);
    ocv_lib::translate(t1,T_v3d);
    
    V3D::refineRelativePose(left, right, Kleft, Kright, R_v3d, T_v3d, 10000.0);
    ocv_lib::translate(R_v3d, R1);
    ocv_lib::translate(T_v3d, t1);

    computeEssentialFromRT(R1,t1,E);
    decomposeEssentialMat(E, R1, R2, t1, t2);


    inFrontNum = selectCorrectRtUsingCherality(R1, R2, t1, t2, m_src, m_tgt, status,R,t);
		
    if ( inFrontNum >= 5) {
      if (consideredNum != inFrontNum) {
      		consideredNum = inFrontNum;
          inFrontNum = 0;

			}
		 } else {
           consideredNum = inFrontNum;
           inFrontNum = 0;
          }
	}
    
	return consideredNum;
 }

double getEssentialRatio(cv::Mat & imagePoint_src, 
                         cv::Mat & imagePoint_tgt, 
                         cv::Mat & K_src, 
                         cv::Mat & K_tgt) 
{
  if (imagePoint_src.cols < 3) {
    std::cerr << "problem in getEssentialRatio input of dim 0" << std::endl;
    return 0.0;
  }

  CvMat imagePoint_src_ = cvMat(3,imagePoint_src.cols, CV_64FC1, imagePoint_src.ptr<double>());
  CvMat imagePoint_tgt_ = cvMat(3,imagePoint_tgt.cols, CV_64FC1, imagePoint_tgt.ptr<double>());

  //CvMat K_src_ = cvMat(3,K_src.cols, CV_64FC1, K_src.ptr<double>());
  //CvMat K_tgt_ = cvMat(3,K_tgt.cols, CV_64FC1, K_tgt.ptr<double>());
  cv::Mat F(3,3,CV_64FC1);
  CvMat F_ = cvMat(3,3,CV_64FC1,F.ptr<double>());
  
  
  cvFindFundamentalMat(&imagePoint_src_, &imagePoint_tgt_, &F_, CV_FM_8POINT);

  cv::Mat E = K_tgt.t() * F * K_src;
   
  cv::SVD svd(E,cv::SVD::MODIFY_A);
  
  return svd.w.at<double>(1) / svd.w.at<double>(0);
  
}

double getEssentialRatio2(cv::Mat & imagePoint_src, 
                         cv::Mat & imagePoint_tgt, 
                         cv::Mat & K_src, 
                         cv::Mat & K_tgt) 
{
  cv::Mat m_src = K_src.inv() * imagePoint_src;
  cv::Mat m_tgt = K_tgt.inv() * imagePoint_tgt;

  CvMat imagePoint_src_ = cvMat(3,imagePoint_src.cols, CV_64FC1, m_src.ptr<double>());
  CvMat imagePoint_tgt_ = cvMat(3,imagePoint_tgt.cols, CV_64FC1, m_tgt.ptr<double>());

  
  cv::Mat E(3,3,CV_64FC1);
  CvMat E_ = cvMat(3,3,CV_64FC1,E.ptr<double>());
  
  
  cvFindFundamentalMat(&imagePoint_src_, &imagePoint_tgt_, &E_, CV_FM_8POINT);

  cv::SVD svd(E,cv::SVD::MODIFY_A);
  
  return svd.w.at<double>(1) / svd.w.at<double>(0);
  
}

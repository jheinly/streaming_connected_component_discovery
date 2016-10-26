#include <estimator/v3d_code/triangulation.h>

#include <estimator/v3d_code/v3d_mviewutilities.h>
using std::string;

bool checkCherality(const cv::Mat & R,
                    const cv::Mat & t,
                    const cv::Mat & M)
{
  cv::Mat M_ = R * M + t;

  return (M_.at<double>(2) > 0.0);
}

double GetCosApicalAngle(const cv::Mat & C1, const cv::Mat & C2, const cv::Mat & pt)
	{
    using namespace cv;
		Mat d1, d2;
		double v;
    d1 = C1 - pt;
    d1 = d1 / norm(d1);
    d2 = C2 - pt;
    d2 = d2 / norm(d2);

		v = d1.dot(d2);

		if(v<0) return 0;

		return v;

	}

double GetApicalAngle(const cv::Mat & C1, const cv::Mat & C2, const cv::Mat & pt) {
#define MY_M_PI 3.14159265358979323846
  return acos(GetCosApicalAngle(C1,C2,pt))*180/MY_M_PI;
#undef MY_M_PI
}


void calculateLinearEq1ForTriangulation(const cv::Mat & P, const cv::Mat & m, cv::Mat & A_) //it is a line in the linear system of A!
{
   A_= (P.row(2) * m.at<double>(0,0) -  P.row(0));
}

void calculateLinearEq2ForTriangulation(const cv::Mat & P, const cv::Mat & m, cv::Mat & A_) //it is a line in the linear system of A!
{
   A_= (P.row(2) * m.at<double>(1,0) - P.row(1));
}

void solveTriangulationLinearSystem(cv::Mat & A, cv::Mat & M) {
  //[U, W, V] = svd(A);
  cv::SVD svd(A,cv::SVD::MODIFY_A);

  //M = V(1:3, 4) / V(4,4);
  M.at<double>(0,0) = svd.vt.at<double>(3,0) / svd.vt.at<double>(3,3);
  M.at<double>(1,0) = svd.vt.at<double>(3,1) / svd.vt.at<double>(3,3);
  M.at<double>(2,0) = svd.vt.at<double>(3,2) / svd.vt.at<double>(3,3);
}

void reconstruct3dPoint(const cv::Mat & m1, const cv::Mat & m2, const cv::Mat & P2, cv::Mat &  M) {
  using namespace cv;

  Mat A(4,4, CV_64FC1,Scalar(0.0));
  //A(1,:) = x(1) * P1(3,:) - P1(1,:);

  A.at<double>(0,0) = -1.0;
  A.at<double>(0,2) = m1.at<double>(0,0);

  //A(2,:) = x(2) * P1(3,:) - P1(2,:);
  A.at<double>(1,1) = -1.0;
  A.at<double>(1,2) = m1.at<double>(1,0);

  //A(3,:) = x_(1) * P2(3,:) - P2(1,:);
  //A.row(2) = (P2.row(2) * m2.at<double>(0,0) -  P2.row(0));
  cv::Mat A_row = A.row(2);
  calculateLinearEq1ForTriangulation(P2, m2, A_row);

  //A(4,:) = x_(2) * P2(3,:) - P2(2,:);
  //  A.row(3) = (P2.row(2) * m2.at<double>(1,0) - P2.row(1));
  A_row = A.row(3);
  calculateLinearEq2ForTriangulation(P2, m2, A_row);


  solveTriangulationLinearSystem(A, M);


}

double reconstruct3dPoint(const vector<cv::Mat> & ps,
                          const vector<cv::Mat> & Ks,
                          const vector<cv::Mat> & dists,
                          const vector<cv::Mat> & Rs,
                          const vector<cv::Mat> & ts,
                          cv::Mat & M)
{
  int nViews = static_cast<int>(ps.size());
  double m_[2];
  cv::Mat m(2,1,CV_64FC1,m_);
  //allocate linear system
  cv::Mat A(2*nViews, 4, CV_64FC1, cv::Scalar(0.0));


  for (int viewIdx=0; viewIdx < nViews; ++viewIdx) {


    cv::Mat P = PfromRt(Rs[viewIdx], ts[viewIdx]); //todo check for efficiency

    ocv_lib::undistortPoints( ps[viewIdx], Ks[viewIdx], dists[viewIdx], m); //todo check for efficiency
    cv::Mat A_row = A.row(2*viewIdx);
    calculateLinearEq1ForTriangulation( P, m , A_row);
    A_row = A.row(2*viewIdx+1);
    calculateLinearEq2ForTriangulation( P, m ,A_row);


   /* std::stringstream  ss;
    ss << viewIdx;

    saveMatForMatlab(ps[viewIdx], string("p_") + ss.str());
    saveMatForMatlab(m, string("m_") + ss.str());
    saveMatForMatlab(Ks[viewIdx], string("K_") + ss.str());
    saveMatForMatlab(Rs[viewIdx], string("R_") + ss.str());
    saveMatForMatlab(ts[viewIdx], string("t_") + ss.str());*/

  }
  solveTriangulationLinearSystem(A, M);

  return getMeanSquaredRerojectionError(ps, Ks, dists, Rs, ts, M);
}


size_t cheralityCheck(const cv::Mat & R, const cv::Mat & t,
		      const cv::Mat & m1, const cv::Mat & m2,
		      std::vector<bool> & status) {
               //mX = inv(K) * pX;
	using namespace cv;

	size_t numberOfPoints = status.size();
	size_t numberOfValidPoint = 0;
	Mat ppvt(3,3,CV_64FC1);
    Mat E(3,3,CV_64FC1);
	Mat ppvRm1I(3,3,CV_64FC1);

	ppv(t,ppvt); // ppvt = ppv(t)
    E = ppvt * R; // compute essential matrix from R and t

	//main loop which conditianaly the second condition
    //break the loop when we are sure that 50% bare cannot be reached with the remaining points maybe using an additional parameter.
	for (size_t mIdx = 0; mIdx < numberOfPoints; ++mIdx) {
		if (status[mIdx]) {

			Mat ppvtm2I = ppvt * m2.col(static_cast<int>(mIdx));
			Mat Em1I = E * m1.col(static_cast<int>(mIdx));

			// (ppv(t) * m2(:,i))' * E * m1(:,i) > 0.0
			double z_MI = ppvtm2I.dot(Em1I);

			if (z_MI > 0.0) {
              //(ppv(R * m1(:,i)) * m2(:,i))' * ppv(R * m1(:,i)) * t > 0.0

				Mat Rm1I = R * m1.col(static_cast<int>(mIdx)); // Rm1 = R * m1
				ppv(Rm1I, ppvRm1I); //ppvRmI1 = ppv(R * m1(:, mXI))

				Mat ppvRm1I_m2I = ppvRm1I * m2.col(static_cast<int>(mIdx)); // ppvRmI1_m2I = ppv(R *m1(:,mXI)) * m2(:,mXI)
				Mat ppvRm1I_t = ppvRm1I * t; // ppvRmI1_t = ppv(R * m1(:,mXI)) * t

                z_MI = ppvRm1I_m2I.dot(ppvRm1I_t);
                status[mIdx]= (z_MI > 0.0);

				if (status[mIdx])
					++numberOfValidPoint;
			} else
              status[mIdx] = false;
		}
	}


	return numberOfValidPoint;
}

double computeBestAA(const cv::Mat & M,
                      const vector<cv::Mat > & Rs,
                      const vector<cv::Mat > & ts,
                      const vector<int> & imageIndexes)
{
  using namespace cv;


  size_t nViews = ts.size();
  vector<cv::Mat> Cs(nViews);
  for (size_t idx=0; idx < nViews; ++idx)
      ocv_lib::getCameraLocation(Rs[idx], ts[idx], Cs[idx]);


  double bestAA = 0.0;
  for (size_t i = 0; i < nViews-1; ++i)

    for (size_t j = i+1; j < nViews; ++j)
      if ( imageIndexes[i] != imageIndexes[j] ) {
        //the camera need to be diffent for the thing to work.
        //it is questionable that a close loop like this is valid but it usually means that there is bellow a 0.1px between the point which might not be a big numerical problem

      double r = GetApicalAngle(Cs[i], Cs[j], M);

      if (r > bestAA)
        bestAA = r;
      }
  return bestAA;
}

double computeBestRoundness(const vector<cv::Mat > &pts,
                            const vector<cv::Mat > &Ks,
                            const vector<cv::Mat > &dists,
                            const vector<cv::Mat > & Rs,
                            const vector<cv::Mat > & ts,
                            const vector<size_t> & imageIndexes)
{
  using namespace cv;

  double sigma = 1.0;
  double mi_[2], mj_[2];
  Mat mi(2,1,CV_64FC1,mi_);
  Mat mj(2,1,CV_64FC1,mj_);



  vector<V3D::Matrix3x4d> projections(2);

  vector<V3D::PointMeasurement> measurements(2);
  measurements[0].view=0;
  measurements[1].view=1;

  double bestRoundness = 0.0;
  for (size_t i = 0; i < pts.size()-1; ++i)
  {
    for (int row=0; row<3; ++row) {
      //copy translation
       projections[0][row][3] = ts[i].at<double>(row);
        for (int col=0; col<3; ++col)
          projections[0][row][col] = Rs[i].at<double>(row,col);
    }

    ocv_lib::undistortPoints(pts[i], Ks[i], dists[i], mi);
    measurements[0].pos[0] = static_cast<float>(mi_[0]);
    measurements[0].pos[1] = static_cast<float>(mi_[1]);

    for (size_t j = i+1; j < pts.size(); ++j) {
      if ( imageIndexes[i] != imageIndexes[j] ) {
        //the camera need to be diffent for the thing to work.
        //it is questionable that a close loop like this is valid but it usually means that there is bellow a 0.1px between the point which might not be a big numerical problem
        for (int row=0; row<3; ++row) {
           //copy translation
           projections[1][row][3] = ts[j].at<double>(row);
           for (int col=0; col<3; ++col)
              projections[1][row][col] = Rs[j].at<double>(row,col);
        }



      ocv_lib::undistortPoints(pts[j], Ks[j], dists[j], mj);

      measurements[1].pos[0] = static_cast<float>(mj_[0]);
      measurements[1].pos[1] = static_cast<float>(mj_[1]);

      double r = computeIntersectionRoundness(projections, measurements, sigma);
      if (r > bestRoundness)
        bestRoundness = r;
      }
    }
  }

  return bestRoundness;
}



size_t getNumberOfStableTriangulationBasedOnRoundness(const cv::Mat & p1, const cv::Mat & p2,
                                                const cv::Mat & K1, const cv::Mat & K2,
                                                const cv::Mat & dist1, const cv::Mat & dist2,
                                                const cv::Mat & P, std::vector<bool> & status,
                                                double threshold)
{
  using namespace cv;

  double sigma = 1.0; /// K1.at<double>(0,0) / K2.at<double>(0,0); //this is from the beders paper everything as to be normalize to the focal plane.
  double m1_[2], m2_[2];
  Mat m1(2,1,CV_64FC1,m1_);
  Mat m2(2,1,CV_64FC1,m2_);

  V3D::Matrix3x4d P0_v3d,P1_v3d;
  P0_v3d[0][0] = P0_v3d[1][1] = P0_v3d[2][2] = 1.0;

  P0_v3d[0][1] = P0_v3d[0][2] = P0_v3d[0][3] = 0.0;
  P0_v3d[1][0] = P0_v3d[1][2] = P0_v3d[1][3] = 0.0;
  P0_v3d[2][0] = P0_v3d[2][1] = P0_v3d[2][3] = 0.0;

  for (int i=0; i<3; ++i)
    for (int j=0; j<4; ++j)
      P1_v3d[i][j] = P.at<double>(i,j);

  vector<V3D::Matrix3x4d> projections(2);
  projections[0] = P0_v3d;
  projections[1] = P1_v3d;


  size_t retVal = 0;

  vector<V3D::PointMeasurement> measurements(2);
  measurements[0].view=0;
  measurements[1].view=1;
  for (int ptsIdx=0; ptsIdx < p1.cols; ++ptsIdx)
  {


    ocv_lib::undistortPoints(p1.col(ptsIdx), K1, dist1, m1);
    ocv_lib::undistortPoints(p2.col(ptsIdx), K2, dist2, m2);

    measurements[0].pos[0] = static_cast<float>(m1.at<double>(0));
    measurements[0].pos[1] = static_cast<float>(m1.at<double>(1));
    measurements[1].pos[0] = static_cast<float>(m2.at<double>(0));
    measurements[1].pos[1] = static_cast<float>(m2.at<double>(1));

    double r = computeIntersectionRoundness(projections, measurements, sigma);

    if (status[ptsIdx] = (r > threshold))
        ++retVal;
  }
  return retVal;

}

double getAverageApicalAngleGetStability(const cv::Mat & p1, const cv::Mat & p2,
                                          const cv::Mat & K1, const cv::Mat & K2,
                                          const cv::Mat & dist1, const cv::Mat & dist2,
                                          const cv::Mat & P, std::vector<bool> & status,
                                          const double min_angle_stability_threshold,
                                          size_t & numOfStablePoint)
{
	using namespace cv;
    numOfStablePoint = 0;
    double averageAngle = 0;
    size_t numberPoints = p1.cols;

    double m1_[2], m2_[2], M_[3];
    Mat M(3,1,CV_64FC1,M_);
    Mat m1(2,1,CV_64FC1,m1_);
    Mat m2(2,1,CV_64FC1,m2_);

    Mat C1(3,1,CV_64FC1,Scalar(0.0));
    Mat C2;
    ocv_lib::getCameraLocation(P, C2);

    /*saveMatForMatlab(P, "P");
    saveMatForMatlab(K1, "K1");
    saveMatForMatlab(K2, "K2");

    saveMatForMatlab(p1, "p1");
    saveMatForMatlab(p2, "p2");*/

    for (size_t ptsIdx=0; ptsIdx < numberPoints; ++ptsIdx) {
      ocv_lib::undistortPoints(p1.col(static_cast<int>(ptsIdx)), K1, dist1, m1);
      ocv_lib::undistortPoints(p2.col(static_cast<int>(ptsIdx)), K2, dist2, m2);

      reconstruct3dPoint(m1, m2, P, M);
      double currentAA = GetApicalAngle(C1, C2, M);
      averageAngle += currentAA;

      if (status[ptsIdx] = ( currentAA >= min_angle_stability_threshold))
        ++numOfStablePoint;

    }
    averageAngle /= static_cast<double>(numberPoints);

    return averageAngle;
}

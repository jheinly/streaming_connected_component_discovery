// -*- C++ -*-

#ifndef V3D_POSE_UTILITIES_H
#define V3D_POSE_UTILITIES_H

#include <vector>
#define _USE_MATH_DEFINES
#include <estimator/v3d_code/v3d_linear.h>

namespace V3D
{

   inline Matrix3x3d
   computeEssentialFromRelativePose(Matrix3x3d const& R, Vector3d const& T)
   {
      Matrix3x3d Tx, E;
      makeCrossProductMatrix(T, Tx);
      multiply_A_B(Tx, R, E);
      return E;
   }

   //! \brief Multiply a given absolute orientation and a relative orientation
   //! to obtain the combined absolute orientation. An additional scale factor
   //! is required. Note: R and T are always provided in the [R|T] notation!
   inline void
   mergeOrientations(Matrix3x3d const& R_abs, Vector3d const& T_abs,
                     Matrix3x3d const& R_rel, Vector3d const& T_rel,
                     double scale,
                     Matrix3x3d& R_merged, Vector3d& T_merged)
   {
      // M1 = [ R1  t1 ]  M2 = [ R2  t2 ]
      //      [ 0   1  ]       [ 0   1  ]
      // ==> M3 = M2 * M1 =  [ R2*R1 | R2*t1 + scale*t2 ]
      R_merged = R_rel * R_abs;
      T_merged = R_rel*T_abs + scale*T_rel;
   }

   inline Matrix3x4d
   mergeOrientations(Matrix3x4d const& RT_abs, Matrix3x4d const& RT_rel, double scale)
   {
      Matrix3x3d R_abs, R_rel, R_merged;
      Vector3d   T_abs, T_rel, T_merged;
      copyMatrixSlice(RT_abs, 0, 0, 3, 3, R_abs, 0, 0);
      copyMatrixSlice(RT_rel, 0, 0, 3, 3, R_rel, 0, 0);
      RT_abs.getColumnSlice(0, 3, 3, T_abs);
      RT_rel.getColumnSlice(0, 3, 3, T_rel);
      mergeOrientations(R_abs, T_abs, R_rel, T_rel, scale, R_merged, T_merged);

      Matrix3x4d res;
      copyMatrixSlice(R_merged, 0, 0, 3, 3, res, 0, 0);
      res.setColumnSlice(0, 3, 3, T_merged);
      return res;
   }

   //! Calculate the relative orientation between two views with known absolute orientation.
   //! Note: R and T are always understood as in the [R|T] notation!
   inline void
   getRelativeOrientation(Matrix3x3d const& R1, Vector3d const& T1,
                          Matrix3x3d const& R2, Vector3d const& T2,
                          Matrix3x3d& R_rel, Vector3d& T_rel)
   {
      R_rel = R2 * R1.transposed();
      T_rel = T2 - R_rel*T1;
   }

   inline Matrix3x4d
   getRelativeOrientation(Matrix3x4d const& RT1, Matrix3x4d const& RT2)
   {
      Matrix3x3d R1, R2, R_rel;
      Vector3d   T1, T2, T_rel;
      copyMatrixSlice(RT1, 0, 0, 3, 3, R1, 0, 0);
      copyMatrixSlice(RT2, 0, 0, 3, 3, R2, 0, 0);
      RT1.getColumnSlice(0, 3, 3, T1);
      RT2.getColumnSlice(0, 3, 3, T2);
      getRelativeOrientation(R1, T1, R2, T2, R_rel, T_rel);

      Matrix3x4d res;
      copyMatrixSlice(R_rel, 0, 0, 3, 3, res, 0, 0);
      res.setColumnSlice(0, 3, 3, T_rel);
      return res;
   }

//**********************************************************************

   void computeAbsolutePose3Point(Vector2f const& p1, Vector2f const& p2, Vector2f const& p3, // Normalized image points
                                  Vector3d const& X1, Vector3d const& X2, Vector3d const& X3, // corresponding 3D points
                                  std::vector<Matrix3x4d>& RTs);
   void computeAbsolutePose3Point(Vector2f const& p1, Vector2f const& p2, Vector2f const& p3, // Normalized image points
                                  Vector3d const& X1, Vector3d const& X2, Vector3d const& X3, // 3D points
                                  Matrix3x4d *RTs, int maxRTs, int &numRTs);

   void computeAbsolutePose3Point(Vector3d const& p1, Vector3d const& p2, Vector3d const& p3, // Spherical image coordinates points
                             Vector3d const& X1, Vector3d const& X2, Vector3d const& X3, // 3D points
                             vector<Matrix3x4d>& RTs);

   void computeAbsolutePose3Point(Vector3d const& p1, Vector3d const& p2, Vector3d const& p3, // Spherical image coordinates points
                                   Vector3d const& X1, Vector3d const& X2, Vector3d const& X3, // 3D points
                                   Matrix3x4d *RTs, int maxRTs, int &numRTs );

   bool refineAbsolutePose(std::vector<Vector2f> const& normalizedPoints2d, std::vector<Vector3d> const& points3d,
                           double inlierThreshold, Matrix3x3d& R, Vector3d& T);


//**********************************************************************

   bool computeEssentialsFromFiveCorrs(double const x1[5], double const y1[5],
                                       double const x2[5], double const y2[5], std::vector<Matrix3x3d>& Es);

   void computeEssentialSVD(Matrix3x3d const& E, Matrix3x3d& U, Matrix3x3d& V);

   int getExactConfiguration(Matrix3x3d const& E, Matrix3x3d const& U, Matrix3x3d const& V,
                             double x1, double y1, double x2, double y2);

   void relativePoseFromEssential(Matrix3x3d const& U, Matrix3x3d const& V,
                                  int configuration, Matrix3x3d& R, Vector3d& T);

   //! Checks the cheirality constraint to recover R and t from an essential matrix.
   /*! Return false, if not all correspondences induce the same 3d configuration.
    */
   bool relativePoseFromEssential(Matrix3x3d const& E, int nCorrs,
                                  double const * x1, double const * y1,
                                  double const * x2, double const * y2,
                                  Matrix3x3d& R, Vector3d& T);

   size_t robustRelativePoseFromEssential(
							Matrix3x3d const& E, size_t nCorrs,
                             double const * x1, double const * y1,
                             double const * x2, double const * y2,
                             Matrix3x3d& R, Vector3d& T,
							 vector<bool> & status);
	 bool refineRelativePose(std::vector<Vector2d> const& left,
                           std::vector<Vector2d> const& right,
                           Matrix3x3d const& Kleft,
                           Matrix3x3d const& Kright,
                           Matrix3x3d& R,
                           Vector3d& T,
                           double sampsonInlierThreshold);

//**********************************************************************

//    enum
//    {
//       V3D_3P1P_METHOD_LEX,
//       V3D_3P1P_METHOD_GRLEX, // Not implemented yet
//    };

//    //! Computes the relative pose between two camera rigs using 1 3D point and 3 image correspondences.
//    /*! This routines assumes that the image features (rays) are rotated such that
//     * the 3D point X at time 1 is [0 0 1] and the 3D point Y at time 2 is [0 0 Y3].
//     * Doubles and long doubles are supported types for Num.
//     * Call explicitly e.g. as computeRelativePose_3Plus1Point<double>(...).
//     */
//    template <typename Num>
//    bool computeRelativePose_3Plus1Point(double const p1[3], double const p2[3],
//                                         double const q1[3], double const q2[3],
//                                         double Y3,
//                                         std::vector<Matrix3x3d>& Rs, std::vector<Vector3d>& Ts,
//                                         int method = V3D_3P1P_METHOD_LEX);

   enum
   {
      V3D_2_1P1P_METHOD_LINEARIZED,
      V3D_2_1P1P_METHOD_LINEARIZED_MRP,
      V3D_2_1P1P_METHOD_QUADRATIC_MRP,
      V3D_2_1P1P_METHOD_REFINE_NONLINEAR,
      V3D_2_1P1P_METHOD_REFINE_MODIFIED_RODRIGUES,
      V3D_2_1P1P_METHOD_REFINE_QUATERNION
   };

   //! Computes the relative pose between two camera rigs using 1 3D point and 2+1 image correspondences.
   /*! This routines assumes that the image features (rays) are rotated such that
    * the 3D point X at time 1 is [0 0 1] and the 3D point Y at time 2 is [0 0 Y3].
    * Doubles and long doubles are supported types for Num.
    * Call explicitly e.g. as computeRelativePose_2_1Plus1Point<double>(...).
    */
   template <typename Num>
   bool computeRelativePose_2_1Plus1Point(Vector2d const& pL1, Vector2d const& pL2, Vector2d const& pR,
                                          Vector2d const& qL1, Vector2d const& qL2, Vector2d const& qR,
                                          Vector3d const& X, Vector3d const& Y, Vector3d const& B,
                                          std::vector<Matrix3x3d>& Rs, std::vector<Vector3d>& Ts,
                                          int method = V3D_2_1P1P_METHOD_QUADRATIC_MRP);

//**********************************************************************

   inline Vector3d
   convertECEF_ToWGS84_LLA(Vector3d const& pos)
   {
      // WGS84 ellipsoid constants:
      double const a = 6378137;
      double const e = 8.1819190842622e-2;
      double const pi = 3.141592653589793;

      double const a2 = a*a;
      double const e2 = e*e;

      double const x = pos[0];
      double const y = pos[1];
      double const z = pos[2];

      double const b = sqrt(a2 * (1-e2));
      double const ep = sqrt((a2-b*b)/(b*b));
      double const p = sqrt(x*x + y*y);
      double const th = atan2(a*z, b*p);
      double       lon = atan2(y, x);

      double const sin_th = sin(th);
      double const cos_th = cos(th);

      double const lat = atan2(z+ep*ep*b*sin_th*sin_th*sin_th,
                               p-e2*a*cos_th*cos_th*cos_th);
      double const s   = sin(lat);
      double const N   = a / sqrt(1-e2*s*s);
      double       alt = p / cos(lat) - N;

      // return lon in range [0,2*pi)
      lon = fmod(lon, 2*pi);

      // correct for numerical instability in altitude near exact poles:
      // (after this correction, error is about 2 millimeters, which is about
      // the same as the numerical precision of the overall function)
      if (fabs(x) < 1 && fabs(y) < 1) alt = fabs(z) - b;

      return makeVector3<double>(lat, lon, alt);
   }

   inline Vector3d
   convertWGS84_LLA_To_ECEF(Vector3d const& lla)
   {
      // WGS84 ellipsoid constants:
      double const a = 6378137;
      double const e = 8.1819190842622e-2;

      double const e2 = e*e;

      double const lat = lla[0];
      double const lon = lla[1];
      double const alt = lla[2];

      double const sinLat = sin(lat);
      double const cosLat = cos(lat);
      double const sinLon = sin(lon);
      double const cosLon = cos(lon);

      // intermediate calculation
      // (prime vertical radius of curvature)
      double const N = a / sqrt(1 - e2 * sinLat*sinLat);

      // results:
      double const x = (N+alt) * cosLat * cosLon;
      double const y = (N+alt) * cosLat * sinLon;
      double const z = ((1-e2) * N + alt) * sinLat;

      return makeVector3<double>(x, y, z);
   }

   //! @brief Get the rotation matrix to rotate ECEF positions into a local north-east-up
   //! position with respect to the given (lat, lon, alt) tripel.
   template <typename Mat>
   inline void
   getECEF_To_LLA_LocalTangentPlaneRotation(Vector3d const& lla, Mat& R)
   {
      R.newsize(3, 3);

      double const lat = lla[0];
      double const lon = lla[1];
      double const alt = lla[2];

      double const sinLat = sin(lat);
      double const cosLat = cos(lat);
      double const sinLon = sin(lon);
      double const cosLon = cos(lon);

      R(1, 1) = -sinLat*cosLon; R(1, 2) = -sinLat*sinLon; R(1, 3) = cosLat;
      R(2, 1) = -sinLon;        R(2, 2) = cosLon;         R(2, 3) = 0.0;
      R(3, 1) =  cosLat*cosLon; R(3, 2) = cosLat*sinLon;  R(3, 3) = sinLat;
   }

   //! @brief Get the rotation matrix to rotate ECEF positions into a local east-north-up (ENU)
   //! position with respect to the given (lat, lon, alt) tripel.
   template <typename Mat>
   inline void
   getECEF_To_ENU_LocalTangentPlaneRotation(Vector3d const& lla, Mat& R)
   {
      R.newsize(3, 3);

      double const lat = lla[0];
      double const lon = lla[1];
      double const alt = lla[2];

      double const sinLat = sin(lat);
      double const cosLat = cos(lat);
      double const sinLon = sin(lon);
      double const cosLon = cos(lon);

      R(1, 1) = -sinLon;        R(1, 2) =  cosLon;        R(1, 3) = 0.0;
      R(2, 1) = -sinLat*cosLon; R(2, 2) = -sinLat*sinLon; R(2, 3) = cosLat;
      R(3, 1) =  cosLat*cosLon; R(3, 2) =  cosLat*sinLon; R(3, 3) = sinLat;
   }

} // end namespace V3D

#endif

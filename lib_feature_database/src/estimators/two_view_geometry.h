// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#ifndef COLMAP_SRC_ESTIMATORS_TWO_VIEW_GEOMETRY_H_
#define COLMAP_SRC_ESTIMATORS_TWO_VIEW_GEOMETRY_H_

#if 0 // Disabled as this code is not necessary for the feature database.
#include "optim/ransac.h"
#include "util/verify.h"
#endif // Disabled as this code is not necessary for the feature database.

namespace colmap {

#if 0 // Disabled as this code is not necessary for the feature database.
struct TwoViewGeometryOptions {
  TwoViewGeometryOptions()
      : min_num_inliers(15),
        min_E_F_inlier_ratio(0.95),
        max_H_inlier_ratio(0.8),
        min_tri_angle(1),
        watermark_min_inlier_ratio(0.7),
        watermark_border_size(0.1),
        detect_watermark(true) {}

  // Minimum number of inliers for non-degenerate two-view geometry
  int min_num_inliers;

  // In case both cameras are calibrated, the calibration is verified by
  // estimating an essential and fundamental matrix and comparing their
  // fractions of number of inliers. If the essential matrix produces
  // a similar number of inliers (`min_E_F_inlier_ratio * F_num_inliers`),
  // the calibration is assumed to be correct.
  double min_E_F_inlier_ratio;

  // In case an epipolar geometry can be verified, it is checked whether
  // the geometry describes a planar scene or panoramic view (pure rotation)
  // described by a homography. This is a degenerate case, since epipolar
  // geometry is only defined for a moving camera. If the inlier ratio of
  // a homography comes close to the inlier ratio of the epipolar geometry,
  // a planar or panoramic configuration is assumed.
  double max_H_inlier_ratio;

  // Whenever a planar or panoramic (calibrated) configuration is detected, the
  // decision whether the former or latter is true, is decided by determining
  // the median triangulation angle of all 3D points. A small triangulation
  // angle indicates a panoramic view.
  double min_tri_angle;

  // In case of valid two-view geometry, it is checked whether the geometry
  // describes a pure translation in the border region of the image. If more
  // than a certain ratio of inlier points conform with a pure image
  // translation, a watermark is assumed.
  double watermark_min_inlier_ratio;

  // Watermark matches have to be in the border region of the image. The
  // border region is defined as the area around the image borders and
  // is defined as a fraction of the image diagonal.
  double watermark_border_size;

  // Whether to enable watermark detection. A watermark causes a pure
  // translation in the image space with inliers in the border region.
  bool detect_watermark;

  RANSACOptions ransac_options;

  void verify() const {
    VERIFY_MSG(min_num_inliers >= 0,
               "Minimum number of inliers must be greater than or equal to 0");
    VERIFY_MSG(min_E_F_inlier_ratio >= 0 && min_E_F_inlier_ratio <= 1,
               "Minimum E over F inlier ratio must be in range [0, 1]");
    VERIFY_MSG(max_H_inlier_ratio >= 0 && max_H_inlier_ratio <= 1,
               "Maximum H inlier ratio must be in range [0, 1]");
    VERIFY_MSG(min_tri_angle >= 0,
               "Minimum triangulation angle must be greater "
               "than or equal to 0");
    VERIFY_MSG(watermark_min_inlier_ratio >= 0 &&
               watermark_min_inlier_ratio <= 1,
               "Watermark inlier ratio must be in range [0, 1]");
    VERIFY_MSG(watermark_border_size >= 0 && watermark_border_size <= 1,
               "Watermark border size must be in range [0, 1]");
    ransac_options.verify();
  }
};
#endif // Disabled as this code is not necessary for the feature database.

struct TwoViewGeometry {
  enum TWO_VIEW_CONFIG {
    DEGENERATE = 1,
    // Essential matrix
    CALIBRATED = 2,
    // Fundamental matrix
    UNCALIBRATED = 3,
    // Homography, planar scene with baseline
    PLANAR = 4,
    // Homography, pure rotation without baseline
    PANORAMIC = 5,
    // Homography, planar or panoramic
    PLANAR_OR_PANORAMIC = 6,
    // Watermark in the image
    WATERMARK = 7,
  };

  TwoViewGeometry()
      : config(DEGENERATE),
        E(Eigen::Matrix3d::Zero()),
        F(Eigen::Matrix3d::Zero()),
        H(Eigen::Matrix3d::Zero()),
        qvec(Eigen::Vector4d::Zero()),
        tvec(Eigen::Vector3d::Zero()),
        inlier_matches(FeatureMatches(0, 2)),
        tri_angle(0),
        E_num_inliers(0),
        F_num_inliers(0),
        H_num_inliers(0) {}

#if 0 // Disabled as this code is not necessary for the feature database.
  /**
   * Estimate two-view geometry from calibrated or uncalibrated image pair,
   * depending on whether a prior focal length is given or not.
   *
   * @param camera1         Camera of first image.
   * @param keypoints1      Keypoints of first image.
   * @param camera2         Camera of second image.
   * @param keypoints2      Keypoints of second image.
   * @param matches         Feature matches between first and second image.
   * @param options         Two-view geometry estimation options.
   */
  void estimate(const Camera& camera1, const FeatureKeypoints& keypoints1,
                const Camera& camera2, const FeatureKeypoints& keypoints2,
                const FeatureMatches& matches,
                const TwoViewGeometryOptions& options);

  /**
   * Estimate two-view geometry from calibrated image pair.
   *
   * @param camera1         Camera of first image.
   * @param keypoints1      Keypoints of first image.
   * @param camera2         Camera of second image.
   * @param keypoints2      Keypoints of second image.
   * @param matches         Feature matches between first and second image.
   * @param options         Two-view geometry estimation options.
   */
  void estimate_calibrated(const Camera& camera1,
                           const FeatureKeypoints& keypoints1,
                           const Camera& camera2,
                           const FeatureKeypoints& keypoints2,
                           const FeatureMatches& matches,
                           const TwoViewGeometryOptions& options);

  /**
   * Estimate two-view geometry from uncalibrated image pair.
   *
   * @param camera1         Camera of first image.
   * @param keypoints1      Keypoints of first image.
   * @param camera2         Camera of second image.
   * @param keypoints2      Keypoints of second image.
   * @param matches         Feature matches between first and second image.
   * @param options         Two-view geometry estimation options.
   */
  void estimate_uncalibrated(const Camera& camera1,
                             const FeatureKeypoints& keypoints1,
                             const Camera& camera2,
                             const FeatureKeypoints& keypoints2,
                             const FeatureMatches& matches,
                             const TwoViewGeometryOptions& options);

  /**
   * Detect if inlier matches are caused by a watermark.
   *
   * A watermark causes a pure translation in the border are of the image.
   */
  static bool detect_watermark(const Camera& camera1,
                               const std::vector<Eigen::Vector2d>& points1,
                               const Camera& camera2,
                               const std::vector<Eigen::Vector2d>& points2,
                               const size_t num_inliers,
                               const std::vector<bool>& inlier_mask,
                               const TwoViewGeometryOptions& options);
#endif // Disabled as this code is not necessary for the feature database.

  // One of `TWO_VIEW_CONFIG`
  int config;

  // Essential matrix
  Eigen::Matrix3d E;
  // Fundamental matrix
  Eigen::Matrix3d F;
  // Homography
  Eigen::Matrix3d H;

  // Relative poses
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;

  FeatureMatches inlier_matches;

  // Median triangulation angle
  double tri_angle;
  size_t E_num_inliers;
  size_t F_num_inliers;
  size_t H_num_inliers;
};

}  // end namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_TWO_VIEW_GEOMETRY_H_

// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#ifndef COLMAP_SRC_BASE2D_CAMERA_H_
#define COLMAP_SRC_BASE2D_CAMERA_H_

#include "util/types.h"

#include <vector>

namespace colmap {

struct Camera {
  Camera()
      : camera_id(INVALID_CAMERA_ID),
        model(-1),
        width(0),
        height(0),
        prior_focal(false) {}

  void init(const double focal, const size_t width, const size_t height);

  Eigen::Vector2d image2world(const Eigen::Vector2d& image_point) const;
  Eigen::Vector3d image2world_homogenous(const Eigen::Vector2d& image_point)
      const;
  Eigen::Vector2d world2image(const Eigen::Vector2d& world_point) const;

  double image2world_threshold(const double threshold) const;

  bool has_bogus_params(const double min_focal_ratio,
                        const double max_focal_ratio,
                        const double max_extra_param) const;

  double mean_focal() const;

  std::string model_name() const;

  camera_t camera_id;

  // Camera model code
  int model;

  // Image dimensions in pixels
  size_t width;
  size_t height;

  // The camera parameters
  std::vector<double> params;

  // Whether there is a safe prior for the focal length,
  // e.g. manually provided or extracted from EXIF
  bool prior_focal;
};

}  // end namespace colmap

#endif  // COLMAP_SRC_BASE2D_CAMERA_H_

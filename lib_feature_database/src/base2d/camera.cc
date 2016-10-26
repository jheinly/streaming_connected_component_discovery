// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#include "base2d/camera.h"
#include "base3d/camera_models.h"

#include <Eigen/Geometry>

namespace colmap {

void Camera::init(const double focal, const size_t width,
                  const size_t height) {
  VERIFY_MSG(model != -1, "Camera model must be set before initialization");

  this->width = width;
  this->height = height;

  camera_init_params(model, focal, width, height, params);
}

Eigen::Vector2d Camera::image2world(const Eigen::Vector2d& image_point) const {
  return image2world_homogenous(image_point).hnormalized();
}

Eigen::Vector3d Camera::image2world_homogenous(
    const Eigen::Vector2d& image_point) const {
  Eigen::Vector3d world_point;
  camera_image2world(image_point(0), image_point(1), world_point(0),
                     world_point(1), world_point(2), model, params);
  return world_point;
}

Eigen::Vector2d Camera::world2image(const Eigen::Vector2d& world_point) const {
  Eigen::Vector2d image_point;
  camera_world2image(world_point(0), world_point(1), 1.0, image_point(0),
                     image_point(1), model, params);
  return image_point;
}

double Camera::image2world_threshold(const double threshold) const {
  return camera_image2world_threshold(threshold, model, params);
}

bool Camera::has_bogus_params(const double min_focal_ratio,
                              const double max_focal_ratio,
                              const double max_extra_param) const {
  return camera_bogus_params(model, width, height, min_focal_ratio,
                             max_focal_ratio, max_extra_param, params);
}

double Camera::mean_focal() const {
  const auto& focal_idxs = camera_get_focal_idxs(model);
  double focal = 0;

  for (const auto idx : focal_idxs) {
    focal += params[idx];
  }

  return focal / focal_idxs.size();
}

std::string Camera::model_name() const { return camera_code_to_name(model); }

}  // end namespace colmap

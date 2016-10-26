// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#ifndef COLMAP_SRC_BASE3D_CAMERA_MODELS_H_
#define COLMAP_SRC_BASE3D_CAMERA_MODELS_H_

#include "util/verify.h"

#include <cfloat>
#include <vector>
#include <string>

#include <boost/algorithm/string.hpp>

#include <Eigen/Core>

namespace colmap {

/**
 * This file defines several different camera models and arbitrary new camera
 * models can be added by the following steps:
 *
 *  1. Add a new struct in this file which implements all th necessary methods.
 *  2. Define an unique name and code for the camera model and add it to
 *     the struct and update `camera_code_to_name` and
 *     `camera_model_name_to_code`.
 *  3. Add camera model to `CAMERA_MODEL_CASES` macro in this file.
 *  4. Add new template specialization of test case for camera model to
 *     `camera_models_test.cc`.
 *
 * A camera model can have three different types of camera parameters: focal, pp
 * (principal point), extra_params (abberation parameters). The parameter array
 * is split into different groups, so that we can enable or disable the
 * refinement of the individual groups during bundle adjustment. It is up to the
 * camera model to access the parameters correctly (it is free to do so in an
 * arbitrary manner) - the parameters are not accessed from outside.
 *
 * A camera model must have the following methods:
 *
 *  - `bogus_params`: detect bogus camera parameters, e.g. caused by refinement
 *    during bundle-adjustment or bogus initial values.
 *  - `world2image`: transform normalized camera coordinates to image
 *    coordinates (the inverse of `image2world`).
 *  - `image2world`: transform image coordinates to normalized camera
 *    coordinates (the inverse of `world2image`).
 *  - `image2world_threshold`: transform a threshold given in pixels to
 *    normalized units (e.g. useful for reprojection error thresholds).
 *
 * Whenever you specify the camera parameters in a list, they must appear
 * exactly in the order as they are accessed in the defined model struct here.
 *
 * The camera models follow the convenction that the upper left image corner has
 * the coordinate (0, 0), the lower right corner (width, height), i.e. that
 * the upper left pixel center has coordinate (0.5, 0.5) and the lower right
 * pixel center has the coordinate (width - 0.5, height - 0.5).
 */

#define CAMERA_MODEL_CASES                    \
  CAMERA_MODEL_CASE(SimplePinholeCameraModel) \
  CAMERA_MODEL_CASE(PinholeCameraModel)       \
  CAMERA_MODEL_CASE(SimpleRadialCameraModel)  \
  CAMERA_MODEL_CASE(OpenCVCameraModel)        \
  CAMERA_MODEL_CASE(FullOpenCVCameraModel)    \
  CAMERA_MODEL_CASE(CataCameraModel)

#define CAMERA_MODEL_SWITCH_CASES                     \
  CAMERA_MODEL_CASES                                  \
  default:                                            \
    VERIFY_MSG(false, "Camera model does not exist"); \
    break;

// Number of iterations for iterative undistortion, 100 should be enough
// even for complex camera models with high order terms
#define UNDISTORTION_NUM_ITERATIONS 100
#define UNDISTORTION_EPS 1e-10

// The "Curiously Recurring Template Pattern" (CRTP) is used here, so that we
// can reuse some shared functionality between all camera models -
// defined in the BaseCameraModel

template <typename C>
struct BaseCameraModel {
  template <typename T>
  static inline bool bogus_params(const size_t width, const size_t height,
                                  const T min_focal_ratio,
                                  const T max_focal_ratio,
                                  const T max_extra_param,
                                  const std::vector<T>& params) {
    if (bogus_pp(width, height, params)) {
      return true;
    }

    if (bogus_focal(width, height, min_focal_ratio, max_focal_ratio, params)) {
      return true;
    }

    if (bogus_extra_params(max_extra_param, params)) {
      return true;
    }

    return false;
  }

  template <typename T>
  static inline bool bogus_focal(const size_t width, const size_t height,
                                 const T min_focal_ratio,
                                 const T max_focal_ratio,
                                 const std::vector<T>& params) {
    const size_t max_size = std::max(width, height);

    for (const auto& idx : C::focal_idxs) {
      const T focal_ratio = params[idx] / max_size;
      if (focal_ratio < min_focal_ratio || focal_ratio > max_focal_ratio) {
        return true;
      }
    }

    return false;
  }

  template <typename T>
  static inline bool bogus_pp(const size_t width, const size_t height,
                              const std::vector<T>& params) {
    const T cx = params[C::pp_idxs[0]];
    const T cy = params[C::pp_idxs[1]];
    return cx < 0 || cx > width || cy < 0 || cy > height;
  }

  template <typename T>
  static inline bool bogus_extra_params(const T max_extra_param,
                                        const std::vector<T>& params) {
    for (const auto& idx : C::extra_params_idxs) {
      if (std::abs(params[idx]) > max_extra_param) {
        return true;
      }
    }

    return false;
  }

  template <typename T>
  static inline T image2world_threshold(const T threshold, const T params[]) {
    T mean_focal = 0;
    for (const auto& idx : C::focal_idxs) {
      mean_focal += params[idx];
    }
    mean_focal /= C::focal_idxs.size();
    return threshold / mean_focal;
  }

  template <typename T>
  static inline void iterative_undistortion(T& x, T& y, const T params[]) {
    T xx = x;
    T yy = y;
    T dx, dy;
    for (size_t i = 0; i < UNDISTORTION_NUM_ITERATIONS; ++i) {
      C::distortion(xx, yy, dx, dy, params);
      T xx_prev = xx;
      T yy_prev = yy;
      xx = x - dx;
      yy = y - dy;
      if (std::abs(xx_prev - xx) < UNDISTORTION_EPS &&
          std::abs(yy_prev - yy) < UNDISTORTION_EPS) {
        break;
      }
    }

    x = xx;
    y = yy;
  }
};

/**
 * Simple Pinhole camera model.
 *
 * No distortion is assumed. Only focal length and principal point is modeled.
 *
 * Parameter list is expected in the following ordering:
 *
 *   f, cx, cy
 *
 * @see https://en.wikipedia.org/wiki/Pinhole_camera_model
*/
struct SimplePinholeCameraModel
    : public BaseCameraModel<SimplePinholeCameraModel> {
  static const char code = 0;
  static const int num_params = 3;
  static const std::string param_info;
  static const std::vector<int> focal_idxs;
  static const std::vector<int> pp_idxs;
  static const std::vector<int> extra_params_idxs;

  static std::string init_param_info() { return "f, cx, cy"; }

  static std::vector<int> init_focal_idxs() {
    std::vector<int> idxs(1);
    idxs[0] = 0;
    return idxs;
  }

  static std::vector<int> init_pp_idxs() {
    std::vector<int> idxs(2);
    idxs[0] = 1;
    idxs[1] = 2;
    return idxs;
  }

  static std::vector<int> init_extra_params_idxs() {
    std::vector<int> idxs;
    return idxs;
  }

  template <typename T>
  static inline void world2image(const T x, const T y, const T z, T& u, T& v,
                                 const T params[]) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Projection to normalized image plane
    u = x / z;
    v = y / z;

    // No distortion

    // Transform to image coordinates
    u = f * u + c1;
    v = f * v + c2;
  }

  template <typename T>
  static inline void image2world(const T u, const T v, T& x, T& y, T& z,
                                 const T params[]) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    x = (u - c1) / f;
    y = (v - c2) / f;
    z = T(1.0);
  }
};

/**
 * Pinhole camera model.
 *
 * No distortion is assumed. Only focal length and principal point is modeled.
 *
 * Parameter list is expected in the following ordering:
 *
 *    fx, fy, cx, cy
 *
 * @see https://en.wikipedia.org/wiki/Pinhole_camera_model
 */
struct PinholeCameraModel : public BaseCameraModel<PinholeCameraModel> {
  static const char code = 1;
  static const int num_params = 4;
  static const std::string param_info;
  static const std::vector<int> focal_idxs;
  static const std::vector<int> pp_idxs;
  static const std::vector<int> extra_params_idxs;

  static std::string init_param_info() { return "fx, fy, cx, cy"; }

  static std::vector<int> init_focal_idxs() {
    std::vector<int> idxs(2);
    idxs[0] = 0;
    idxs[1] = 1;
    return idxs;
  }

  static std::vector<int> init_pp_idxs() {
    std::vector<int> idxs(2);
    idxs[0] = 2;
    idxs[1] = 3;
    return idxs;
  }

  static std::vector<int> init_extra_params_idxs() {
    std::vector<int> idxs;
    return idxs;
  }

  template <typename T>
  static inline void world2image(const T x, const T y, const T z, T& u, T& v,
                                 const T params[]) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Projection to normalized image plane
    u = x / z;
    v = y / z;

    // No distortion

    // Transform to image coordinates
    u = f1 * u + c1;
    v = f2 * v + c2;
  }

  template <typename T>
  static inline void image2world(const T u, const T v, T& x, T& y, T& z,
                                 const T params[]) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    x = (u - c1) / f1;
    y = (v - c2) / f2;
    z = T(1.0);
  }
};

/**
 * Simple camera model with one focal and one radial distortion parameter.
 *
 * Parameter list is expected in the following ordering:
 *
 *    f, cx, cy, k
 */
struct SimpleRadialCameraModel
    : public BaseCameraModel<SimpleRadialCameraModel> {
  static const char code = 2;
  static const int num_params = 4;
  static const std::string param_info;
  static const std::vector<int> focal_idxs;
  static const std::vector<int> pp_idxs;
  static const std::vector<int> extra_params_idxs;

  static std::string init_param_info() { return "f, cx, cy, k"; }

  static std::vector<int> init_focal_idxs() {
    std::vector<int> idxs(1);
    idxs[0] = 0;
    return idxs;
  }

  static std::vector<int> init_pp_idxs() {
    std::vector<int> idxs(2);
    idxs[0] = 1;
    idxs[1] = 2;
    return idxs;
  }

  static std::vector<int> init_extra_params_idxs() {
    std::vector<int> idxs(1);
    idxs[0] = 3;
    return idxs;
  }

  template <typename T>
  static inline void world2image(const T x, const T y, const T z, T& u, T& v,
                                 const T params[]) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Projection to normalized image plane
    u = x / z;
    v = y / z;

    // Distortion
    T du, dv;
    distortion(u, v, du, dv, &params[3]);
    u += du;
    v += dv;

    // Transform to image coordinates
    u = f * u + c1;
    v = f * v + c2;
  }

  template <typename T>
  static inline void image2world(const T u, const T v, T& x, T& y, T& z,
                                 const T params[]) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Lift points to normalized plane
    x = (u - c1) / f;
    y = (v - c2) / f;
    z = T(1.0);

    iterative_undistortion(x, y, &params[3]);
  }

  template <typename T>
  static inline void distortion(const T u, const T v, T& du, T& dv,
                                const T extra_params[]) {
    const T k = extra_params[0];

    T u2 = u * u;
    T v2 = v * v;
    T r2 = u2 + v2;
    T radial = k * r2;
    du = u * radial;
    dv = v * radial;
  }
};

/**
 * OpenCV camera model.
 *
 * Based on the pinhole camera model. Additionally models radial and
 * tangential distortion (up to 2nd degree of coefficients). Not suitable for
 * large radial distortions of fish-eye cameras.
 *
 * Parameter list is expected in the following ordering:
 *
 *    fx, fy, cx, cy, k1, k2, p1, p2
 *
 * @see
 * http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
 */
struct OpenCVCameraModel : public BaseCameraModel<OpenCVCameraModel> {
  static const char code = 3;
  static const int num_params = 8;
  static const std::string param_info;
  static const std::vector<int> focal_idxs;
  static const std::vector<int> pp_idxs;
  static const std::vector<int> extra_params_idxs;

  static std::string init_param_info() {
    return "fx, fy, cx, cy, k1, k2, p1, p2";
  }

  static std::vector<int> init_focal_idxs() {
    std::vector<int> idxs(2);
    idxs[0] = 0;
    idxs[1] = 1;
    return idxs;
  }

  static std::vector<int> init_pp_idxs() {
    std::vector<int> idxs(2);
    idxs[0] = 2;
    idxs[1] = 3;
    return idxs;
  }

  static std::vector<int> init_extra_params_idxs() {
    std::vector<int> idxs(4);
    idxs[0] = 4;
    idxs[1] = 5;
    idxs[2] = 6;
    idxs[3] = 7;
    return idxs;
  }

  template <typename T>
  static inline void world2image(const T x, const T y, const T z, T& u, T& v,
                                 const T params[]) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Projection to normalized image plane
    u = x / z;
    v = y / z;

    // Distortion
    T du, dv;
    distortion(u, v, du, dv, &params[4]);
    u += du;
    v += dv;

    // Transform to image coordinates
    u = f1 * u + c1;
    v = f2 * v + c2;
  }

  template <typename T>
  static inline void image2world(const T u, const T v, T& x, T& y, T& z,
                                 const T params[]) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    x = (u - c1) / f1;
    y = (v - c2) / f2;
    z = T(1.0);

    iterative_undistortion(x, y, &params[4]);
  }

  template <typename T>
  static inline void distortion(const T u, const T v, T& du, T& dv,
                                const T extra_params[]) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];
    const T p1 = extra_params[2];
    const T p2 = extra_params[3];

    T u2 = u * u;
    T uv = u * v;
    T v2 = v * v;
    T r2 = u2 + v2;
    T radial = k1 * r2 + k2 * r2 * r2;
    du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
    dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
  }
};

/**
 * Full OpenCV camera model.
 *
 * Based on the pinhole camera model. Additionally models radial and
 * tangential distortion.
 *
 * Parameter list is expected in the following ordering:
 *
 *    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
 *
 * @see
 * http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
 */
struct FullOpenCVCameraModel : public BaseCameraModel<FullOpenCVCameraModel> {
  static const char code = 4;
  static const int num_params = 12;
  static const std::string param_info;
  static const std::vector<int> focal_idxs;
  static const std::vector<int> pp_idxs;
  static const std::vector<int> extra_params_idxs;

  static std::string init_param_info() {
    return "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6";
  }

  static std::vector<int> init_focal_idxs() {
    std::vector<int> idxs(2);
    idxs[0] = 0;
    idxs[1] = 1;
    return idxs;
  }

  static std::vector<int> init_pp_idxs() {
    std::vector<int> idxs(2);
    idxs[0] = 2;
    idxs[1] = 3;
    return idxs;
  }

  static std::vector<int> init_extra_params_idxs() {
    std::vector<int> idxs(8);
    idxs[0] = 4;
    idxs[1] = 5;
    idxs[2] = 6;
    idxs[3] = 7;
    idxs[4] = 8;
    idxs[5] = 9;
    idxs[6] = 10;
    idxs[7] = 11;
    return idxs;
  }

  template <typename T>
  static inline void world2image(const T x, const T y, const T z, T& u, T& v,
                                 const T params[]) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Projection to normalized image plane
    u = x / z;
    v = y / z;

    // Distortion
    T du, dv;
    distortion(u, v, du, dv, &params[4]);
    u += du;
    v += dv;

    // Transform to image coordinates
    u = f1 * u + c1;
    v = f2 * v + c2;
  }

  template <typename T>
  static inline void image2world(const T u, const T v, T& x, T& y, T& z,
                                 const T params[]) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    x = (u - c1) / f1;
    y = (v - c2) / f2;
    z = T(1.0);

    iterative_undistortion(x, y, &params[4]);
  }

  template <typename T>
  static inline void distortion(const T u, const T v, T& du, T& dv,
                                const T extra_params[]) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];
    const T p1 = extra_params[2];
    const T p2 = extra_params[3];
    const T k3 = extra_params[4];
    const T k4 = extra_params[5];
    const T k5 = extra_params[6];
    const T k6 = extra_params[7];

    T u2 = u * u;
    T uv = u * v;
    T v2 = v * v;
    T r2 = u2 + v2;
    T r4 = r2 * r2;
    T r6 = r4 * r2;
    T radial = (T(1) + k1 * r2 + k2 * r4 + k3 * r6) /
               (T(1) + k4 * r2 + k5 * r4 + k6 * r6);
    du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2) - u;
    dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2) - v;
  }
};

/**
 * Cata camera model (by Christopher Mei).
 *
 * Based on the OpenCV camera model. It also models the distortion
 * introduced by the use of telecentric lenses with paracatadioptric sensors.
 * It is better suited for fisheye cameras while still being simple.
 *
 * This model is an extension to the models presented in:
 *
 *    Joao P. Barreto and Helder Araujo. Issues on the geometry of central
 *    catadioptric image formation. In CVPR, volume 2, pages 422-427, 2001.
 *    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.95.7958
 *
 *    Christopher Geyer and Kostas Daniilidis. A Unifying Theory for Central
 *    Panoramic Systems and Practical Implications. In ECCV, pages 445-461,
 *    2000. http://www.frc.ri.cmu.edu/users/cgeyer/papers/geyer_eccv00.pdf
 *
 * Parameter list is expected in the following ordering:
 *
 *    fx, fy, cx, cy, k1, k2, p1, p2, xi
 *
 * @see http://homepages.laas.fr/~cmei/uploads/Main/projection_model.pdf
 */
struct CataCameraModel : public BaseCameraModel<CataCameraModel> {
  static const char code = 5;
  static const int num_params = 9;
  static const std::string param_info;
  static const std::vector<int> focal_idxs;
  static const std::vector<int> pp_idxs;
  static const std::vector<int> extra_params_idxs;

  static std::string init_param_info() {
    return "fx, fy, cx, cy, k1, k2, p1, p2, xi";
  }

  static std::vector<int> init_focal_idxs() {
    std::vector<int> idxs(2);
    idxs[0] = 0;
    idxs[1] = 1;
    return idxs;
  }

  static std::vector<int> init_pp_idxs() {
    std::vector<int> idxs(2);
    idxs[0] = 2;
    idxs[1] = 3;
    return idxs;
  }

  static std::vector<int> init_extra_params_idxs() {
    std::vector<int> idxs(5);
    idxs[0] = 4;
    idxs[1] = 5;
    idxs[2] = 6;
    idxs[3] = 7;
    idxs[4] = 8;
    return idxs;
  }

  template <typename T>
  static inline void world2image(const T x, const T y, const T z, T& u, T& v,
                                 const T params[]) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];
    const T xi = params[8];

    // Projection to normalized image plane
    T zz = z + xi * sqrt(x * x + y * y + z * z);
    u = x / zz;
    v = y / zz;

    // Distortion
    T du, dv;
    distortion(u, v, du, dv, &params[4]);
    u += du;
    v += dv;

    // Transform to image coordinates
    u = f1 * u + c1;
    v = f2 * v + c2;
  }

  template <typename T>
  static inline void image2world(const T u, const T v, T& x, T& y, T& z,
                                 const T params[]) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];
    const T xi = params[8];

    // Lift points to normalized plane
    x = (u - c1) / f1;
    y = (v - c2) / f2;

    iterative_undistortion(x, y, &params[4]);

    // Lift normalized points to the sphere
    if (xi == 1) {
      z = (1 - x * x - y * y) / 2;
    } else {
      T r2 = x * x + y * y;
      z = 1 - xi * (r2 + T(1)) / (xi + sqrt(T(1) + (T(1) - xi * xi) * r2));
    }
  }

  template <typename T>
  static inline void distortion(const T u, const T v, T& du, T& dv,
                                const T extra_params[]) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];
    const T p1 = extra_params[2];
    const T p2 = extra_params[3];

    T u2 = u * u;
    T uv = u * v;
    T v2 = v * v;
    T r2 = u2 + v2;
    T radial = k1 * r2 + k2 * r2 * r2;
    du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
    dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
  }
};

/**
 * Convert camera name to unique camera model code.
 *
 * @param name         Unique name of camera model.
 *
 * @return             Unique code of camera model.
 */
inline int camera_name_to_code(std::string name) {
  boost::to_upper(name);
  if (name == "SIMPLE_PINHOLE") {
    return SimplePinholeCameraModel::code;
  } else if (name == "PINHOLE") {
    return PinholeCameraModel::code;
  } else if (name == "SIMPLE_RADIAL") {
    return SimpleRadialCameraModel::code;
  } else if (name == "OPENCV") {
    return OpenCVCameraModel::code;
  } else if (name == "FULL_OPENCV") {
    return FullOpenCVCameraModel::code;
  } else if (name == "CATA") {
    return CataCameraModel::code;
  }
  return -1;
}

/**
 * Convert camera code to unique camera model name.
 *
 * @param name         Unique code of camera model.
 *
 * @return             Unique name of camera model.
 */
inline std::string camera_code_to_name(const int code) {
  if (code == SimplePinholeCameraModel::code) {
    return "SIMPLE_PINHOLE";
  } else if (code == PinholeCameraModel::code) {
    return "PINHOLE";
  } else if (code == SimpleRadialCameraModel::code) {
    return "SIMPLE_RADIAL";
  } else if (code == OpenCVCameraModel::code) {
    return "OPENCV";
  } else if (code == FullOpenCVCameraModel::code) {
    return "FULL_OPENCV";
  } else if (code == CataCameraModel::code) {
    return "CATA";
  }
  return "";
}

/**
 * Transform world coordinates in camera coordinate system to image coordinates.
 *
 * This is the inverse of `camera_model_image2world`.
 *
 * @param x, y, z      World coordinates in camera coordinate system.
 * @param u, v         Output image coordinates in pixels.
 * @param model        Unique code of camera model as defined in
 *                     `CAMERA_MODEL_NAME_TO_CODE`.
 * @param params       Array of camera parameters.
 */
inline void camera_world2image(const double x, const double y, const double z,
                               double& u, double& v, const int model,
                               const std::vector<double>& params) {
  switch (model) {
#define CAMERA_MODEL_CASE(CameraModel)                      \
  case CameraModel::code:                                   \
    CameraModel::world2image(x, y, z, u, v, params.data()); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

/**
 * Transform image coordinates to world coordinates in camera coordinate system.
 *
 * This is the inverse of `camera_model_world2image`.
 *
 * @param u, v         Image coordinates in pixels.
 * @param x, y, z      Output world coordinates in camera coordinate system.
 * @param model        Unique code of camera model as defined in
 *                     `CAMERA_MODEL_NAME_TO_CODE`.
 * @param params       Array of camera parameters.
 */
inline void camera_image2world(const double u, const double v, double& x,
                               double& y, double& z, const int model,
                               const std::vector<double>& params) {
  switch (model) {
#define CAMERA_MODEL_CASE(CameraModel)                      \
  case CameraModel::code:                                   \
    CameraModel::image2world(u, v, x, y, z, params.data()); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

/**
 * Initialize camera parameters using given image properties.
 *
 * Initializes all focal length parameters to the same given focal length and
 * sets the principal point to the center of the image plane.
 *
 * @param model        Unique code of camera model as defined in
 *                     `CAMERA_MODEL_NAME_TO_CODE`.
 * @param focal        Focal length, equal for all focal length parameters.
 * @param width        Sensor width of the camera.
 * @param height       Sensor height of the camera.
 * @param params       Array of camera parameters.
 */
inline void camera_init_params(const int model, const double focal,
                               const size_t width, const size_t height,
                               std::vector<double>& params) {
  // Assuming that image measurements are within [0, dim], i.e. that the
  // upper left corner is the (0, 0) coordinate (rather than the center of
  // the upper left pixel). This complies with the default SiftGPU convention.

  switch (model) {
#define CAMERA_MODEL_CASE(CameraModel)                     \
  case CameraModel::code:                                  \
    params.resize(CameraModel::num_params);                \
    for (const int idx : CameraModel::focal_idxs) {        \
      params[idx] = focal;                                 \
    }                                                      \
    params[CameraModel::pp_idxs[0]] = width / 2.0;         \
    params[CameraModel::pp_idxs[1]] = height / 2.0;        \
    for (const int idx : CameraModel::extra_params_idxs) { \
      params[idx] = 0;                                     \
    }                                                      \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

/**
 * Convert camera parameters from and to CSV string.
 */
std::string camera_params_to_string(const std::vector<double>& params);

std::vector<double> camera_params_from_string(const int code,
                                              const std::string& string);

/**
 * Helper functions to dispatch the access to the functions of camera models.
 */

inline double camera_image2world_threshold(const double threshold,
                                           const int model,
                                           const std::vector<double>& params) {
  switch (model) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::code:                                                \
    return CameraModel::image2world_threshold(threshold, params.data()); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return -1;
}

inline std::string camera_param_info(const int model) {
  switch (model) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::code:              \
    return CameraModel::param_info;    \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return "Camera model does not exist.";
}

inline std::vector<int> camera_get_focal_idxs(const int model) {
  switch (model) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::code:              \
    return CameraModel::focal_idxs;    \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return std::vector<int>(0);
}

inline std::vector<int> camera_get_pp_idxs(const int model) {
  switch (model) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::code:              \
    return CameraModel::pp_idxs;       \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return std::vector<int>(0);
}

inline std::vector<int> camera_get_extra_params_idxs(const int model) {
  switch (model) {
#define CAMERA_MODEL_CASE(CameraModel)     \
  case CameraModel::code:                  \
    return CameraModel::extra_params_idxs; \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return std::vector<int>(0);
}

inline bool camera_verify_params(const int model,
                                 const std::vector<double>& params) {
  switch (model) {
#define CAMERA_MODEL_CASE(CameraModel)              \
  case CameraModel::code:                           \
    if (params.size() == CameraModel::num_params) { \
      return true;                                  \
    }                                               \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return false;
}

inline bool camera_bogus_params(const int model, const size_t width,
                                const size_t height,
                                const double min_focal_ratio,
                                const double max_focal_ratio,
                                const double max_extra_param,
                                const std::vector<double>& params) {
  switch (model) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::code:                                              \
    return CameraModel::bogus_params(width, height, min_focal_ratio,   \
                                     max_focal_ratio, max_extra_param, \
                                     params);                          \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return false;
}

}  // end namespace colmap

#endif  // COLMAP_SRC_BASE3D_CAMERA_MODELS_H_

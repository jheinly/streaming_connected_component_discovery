// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#include "base3d/camera_models.h"
#include "util/misc.h"
#include <iomanip>

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// Initialize param_info, focal_idxs, pp_idxs, extra_params_idxs
////////////////////////////////////////////////////////////////////////////////
#define CAMERA_MODEL_CASE(CameraModel)                                        \
  const std::string CameraModel::param_info = CameraModel::init_param_info(); \
  const std::vector<int> CameraModel::focal_idxs =                            \
      CameraModel::init_focal_idxs();                                         \
  const std::vector<int> CameraModel::pp_idxs = CameraModel::init_pp_idxs();  \
  const std::vector<int> CameraModel::extra_params_idxs =                     \
      CameraModel::init_extra_params_idxs();

CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE

std::string camera_params_to_string(const std::vector<double>& params) {
  std::string string;
  for (const double param : params) {
    std::ostringstream param_text;
    param_text << param;
    string += param_text.str() + ", ";
  }
  return string.substr(0, string.length() - 2);
}

std::vector<double> camera_params_from_string(const int code,
                                              const std::string& string) {
  const std::vector<double> params = csv2vector<double>(string);

  if (camera_verify_params(code, params)) {
    return params;
  } else {
    return std::vector<double>(0);
  }
}

}  // end namespace colmap

// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#ifndef COLMAP_SRC_BASE2D_IMAGE_DATA_H_
#define COLMAP_SRC_BASE2D_IMAGE_DATA_H_

#include "util/types.h"

#include <string>
#include <vector>

namespace colmap {

struct ImageData {
  ImageData()
      : image_id(INVALID_IMAGE_ID),
        camera_id(INVALID_CAMERA_ID),
        name(""),
        roll(0),
        pitch(0),
        yaw(0),
        x(0),
        y(0),
        z(0) {}

  image_t image_id;
  camera_t camera_id;

  std::string name;

  double roll;
  double pitch;
  double yaw;

  double x;
  double y;
  double z;
};

}  // end namespace colmap

#endif  // COLMAP_SRC_BASE2D_IMAGE_DATA_H_

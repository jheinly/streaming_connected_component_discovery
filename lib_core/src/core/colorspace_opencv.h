#pragma once
#ifndef COLORSPACE_OPENCV_H
#define COLORSPACE_OPENCV_H

#include <core/colorspace.h>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace core {

namespace colorspace {

inline int convert_to_opencv(const Colorspace colorspace)
{
  switch (colorspace)
  {
    case RGB:
      return CV_8UC3;
    case Grayscale:
      return CV_8UC1;
  }
  std::cerr << "ERROR: unrecognized colorspace" << std::endl;
  return CV_8UC3;
}

} // namespace colorspace

} // namespace core

#endif // COLORSPACE_OPENCV_H

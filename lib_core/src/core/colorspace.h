#pragma once
#ifndef COLORSPACE_H
#define COLORSPACE_H

#include <iostream>

namespace core {

namespace colorspace {

enum Colorspace {RGB, Grayscale};

inline int num_bytes_per_pixel(const Colorspace colorspace)
{
  switch (colorspace)
  {
    case RGB:
      return 3;
    case Grayscale:
      return 1;
  }
  std::cerr << "ERROR: unrecognized colorspace" << std::endl;
  return 0;
}

} // namespace colorspace

} // namespace core

#endif // COLORSPACE_H

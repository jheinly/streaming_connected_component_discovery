#pragma once
#ifndef SIFT_DESCRIPTOR_H
#define SIFT_DESCRIPTOR_H

#include <core/alignment.h>
#include <core/compiler_macros.h>

namespace features {

struct
#if defined VISUAL_STUDIO_VERSION && VISUAL_STUDIO_VERSION > VISUAL_STUDIO_2010
  ALIGN(512)
#endif
  SiftDescriptorFloat
{
  static const int num_floats_per_descriptor = 128;
  static const int num_bytes_per_descriptor = 4 * num_floats_per_descriptor;

  float floats[num_floats_per_descriptor];
};

struct
#if defined VISUAL_STUDIO_VERSION && VISUAL_STUDIO_VERSION > VISUAL_STUDIO_2010
  ALIGN(128)
#endif
  SiftDescriptorUChar
{
  static const int num_uchars_per_descriptor = 128;
  static const int num_bytes_per_descriptor = num_uchars_per_descriptor;

  unsigned char uchars[num_uchars_per_descriptor];
};

} // namespace features

#endif // SIFT_DESCRIPTOR_H

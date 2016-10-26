#pragma once
#ifndef IMAGE_CACHE_ENTRY_H
#define IMAGE_CACHE_ENTRY_H

#include <main_module/streaming_ipc2sfm_types.h>
#include <core/image_dimensions.h>
#include <features/sift_keypoint.h>
#include <features/sift_descriptor.h>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <vector>
#include <string>

struct ImageCacheEntry
{
  image_index_t image_index;
  std::string image_name;
  core::ImageDimensions dimensions;
  float focal;
  int num_features;
  std::vector<features::SiftKeypoint> keypoints;
  std::vector<features::SiftDescriptorUChar> descriptors_uchar;
  std::vector<std::vector<int> > visual_words;
  boost::shared_ptr<boost::mutex> visual_words_mutex;
  float image_scale_factor;

  void concatenate_all_visual_words(
    std::vector<int> & all_visual_words) const;
};

#endif // IMAGE_CACHE_ENTRY_H

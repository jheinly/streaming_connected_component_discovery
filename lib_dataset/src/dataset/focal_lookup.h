#pragma once
#ifndef FOCAL_LOOKUP_H
#define FOCAL_LOOKUP_H

#include <core/image_dimensions.h>
#include <core/signal.h>
#include <boost/unordered_map.hpp>
#include <boost/thread/thread.hpp>
#include <string>

namespace dataset {

class FocalLookup
{
  public:
    struct Result
    {
      Result()
      : focal_in_pixels(0),
        is_focal_from_exif(false)
      {}

      Result(const float focal, bool is_from_exif)
      : focal_in_pixels(focal),
        is_focal_from_exif(is_from_exif)
      {}

      float focal_in_pixels;
      bool is_focal_from_exif;
    };

    explicit FocalLookup();

    ~FocalLookup();

    void load_dimension_list(
      const std::string & dimension_list_filepath);
    void load_focal_list(
      const std::string & focal_list_filepath);

    void load_dimension_list_with_thread(
      const std::string & dimension_list_filepath);
    void load_focal_list_with_thread(
      const std::string & focal_list_filepath);

    void wait_for_thread_to_load_dimension_list();
    void wait_for_thread_to_load_focal_list();

    Result lookup_focal(
      const std::string & image_name,
      const core::ImageDimensions & image_dimensions,
      const float image_scale_factor = 0.0f) const;

    int num_dimension_entries() const
    { return static_cast<int>(m_image_name_to_dimensions_map.size()); }

    int num_focal_entries() const
    { return static_cast<int>(m_image_name_to_focal_map.size()); }

  private:
    Result compute_default_focal_and_create_result(
      const core::ImageDimensions & image_dimensions) const;
    Result verify_focal_and_create_result(
      const core::ImageDimensions & image_dimensions,
      const float focal,
      const bool is_focal_from_exif) const;

    boost::unordered_map<std::string, core::ImageDimensions> m_image_name_to_dimensions_map;
    boost::unordered_map<std::string, float> m_image_name_to_focal_map;

    boost::thread m_dimension_list_thread;
    boost::thread m_focal_list_thread;
};

} // namespace dataset

#endif // FOCAL_LOOKUP_H

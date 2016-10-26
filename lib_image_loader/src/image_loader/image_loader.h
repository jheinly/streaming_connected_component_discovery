#pragma once
#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <core/colorspace.h>
#include <image_resizer/image_resizer.h>
#include <jpeg_parser/jpeg_parser.h>
#include <iostream>
#include <string>

namespace image_loader {

class ImageLoader
{
public:
  ImageLoader(
    const core::colorspace::Colorspace output_image_colorspace,
    const image_resizer::ImageResizer::ResizeMode output_image_resize_mode,
    const int resized_image_dimension,
    const int minimum_allowed_image_dimension = 1,
    const float maximum_allowed_aspect_ratio = 0.0f,
    const int maximum_loaded_image_dimension_hint = 0);

  ~ImageLoader();

  bool start_loading_jpeg_from_file(
    const std::string & jpeg_filename);

  bool start_loading_jpeg_from_data(
    const unsigned char * const jpeg_data,
    const int jpeg_data_size);

  image_resizer::ImageResizer::ResizeResult finish_loading_jpeg(
    unsigned char * const preallocated_image_data,
    const int preallocated_image_data_size);

private:
  ImageLoader(const ImageLoader &);
  ImageLoader & operator=(const ImageLoader &);

  class LoadFromFile
  {
  public:
    LoadFromFile(const std::string & jpeg_filename)
      : m_jpeg_filename(jpeg_filename)
    {}

    jpeg_parser::ParseStatus operator()(
      unsigned char * const preallocated_image_data,
      const int preallocated_image_data_size,
      int & width_out,
      int & height_out,
      int & bytes_per_pixel_out,
      core::colorspace::Colorspace output_colorspace,
      const int downsample_to_minimum_dimension,
      float * applied_scale_factor_out) const;

  private:
    LoadFromFile & operator=(const LoadFromFile &);

    const std::string & m_jpeg_filename;
  };

  class LoadFromData
  {
  public:
    LoadFromData(
      const unsigned char * const jpeg_data,
      const int jpeg_data_size)
      : m_jpeg_data(jpeg_data), m_jpeg_data_size(jpeg_data_size)
    {}

    jpeg_parser::ParseStatus operator()(
      unsigned char * const preallocated_image_data,
      const int preallocated_image_data_size,
      int & width_out,
      int & height_out,
      int & bytes_per_pixel_out,
      core::colorspace::Colorspace output_colorspace,
      const int downsample_to_minimum_dimension,
      float * applied_scale_factor_out) const;

  private:
    LoadFromData & operator=(const LoadFromData &);

    const unsigned char * const m_jpeg_data;
    const int m_jpeg_data_size;
  };

  template<typename LoadFromSource>
  bool start_loading_helper(const LoadFromSource & load_from_source);

  core::colorspace::Colorspace m_output_image_colorspace;
  image_resizer::ImageResizer m_image_resizer;
  int m_minimum_allowed_image_dimension;
  float m_maximum_allowed_aspect_ratio;
  unsigned char * m_loaded_image_data;
  int m_loaded_image_data_size;
  int m_loaded_image_width;
  int m_loaded_image_height;
  float m_applied_scale_factor;
};

template<typename LoadFromSource>
bool ImageLoader::start_loading_helper(const LoadFromSource & load_from_source)
{
  m_loaded_image_width = 0;
  m_loaded_image_height = 0;
  int loaded_image_bytes_per_pixel = 0;
  jpeg_parser::ParseStatus parse_status = load_from_source(
    m_loaded_image_data,
    m_loaded_image_data_size,
    m_loaded_image_width,
    m_loaded_image_height,
    loaded_image_bytes_per_pixel,
    m_output_image_colorspace,
    m_image_resizer.resized_image_dimension(),
    &m_applied_scale_factor);

  // If the preallocated buffer was not large enough, resize it and attempt to
  // load the JPEG again.
  if (parse_status == jpeg_parser::ResizeRequired)
  {
    int new_num_pixels = m_loaded_image_width * m_loaded_image_height;

    // If the image is greater than 50 megapixels, skip it.
    if (new_num_pixels > 50000000)
    {
      return false;
    }

    delete[] m_loaded_image_data;
    m_loaded_image_data = NULL;

    m_loaded_image_data_size =
      core::colorspace::num_bytes_per_pixel(m_output_image_colorspace) * new_num_pixels;
    m_loaded_image_data = new unsigned char[m_loaded_image_data_size];

    if (m_loaded_image_data == NULL)
    {
      std::cerr << "ERROR: ImageLoader - failed to allocate new image data" << std::endl;
      return false;
    }

    // Attempt to load the JPEG again.
    parse_status = load_from_source(
      m_loaded_image_data,
      m_loaded_image_data_size,
      m_loaded_image_width,
      m_loaded_image_height,
      loaded_image_bytes_per_pixel,
      m_output_image_colorspace,
      m_image_resizer.resized_image_dimension(),
      &m_applied_scale_factor);
  } // end (parse_status == jpeg_parser::ResizeRequired)

  if (parse_status == jpeg_parser::Failed)
  {
    return false;
  }

  // TODO(jheinly) - Add the ability to decode the JPEG header without the rest of the image
  //        data so that the minimum allowed image dimension and maximum allowed
  //        aspect ratio tests can be applied before parsing the JPEG image data.

  // If the image is too small, skip it.
  if (m_loaded_image_width < m_minimum_allowed_image_dimension ||
    m_loaded_image_height < m_minimum_allowed_image_dimension)
  {
    return false;
  }

  // Check if the maximum aspect ratio test has been enabled.
  if (m_maximum_allowed_aspect_ratio > 0)
  {
    // If the image has an aspect ratio that is too large (for example, it
    // could be a panorama), skip it.
    const float aspect_ratio = (m_loaded_image_width > m_loaded_image_height) ?
      (float(m_loaded_image_width) / float(m_loaded_image_height)) :
      (float(m_loaded_image_height) / float(m_loaded_image_width));
    if (aspect_ratio > m_maximum_allowed_aspect_ratio)
    {
      return false;
    }
  }

  return true;
}

} // namespace image_loader

#endif // IMAGE_LOADER_H

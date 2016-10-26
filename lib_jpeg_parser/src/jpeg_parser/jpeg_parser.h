#pragma once
#ifndef JPEG_PARSER_H
#define JPEG_PARSER_H

#include <string>
#include <vector>
#include <core/colorspace.h>

// NOTE: If you get the error, "unsupported color conversion request", it means
//       that the original image was in the CMYK color format.

namespace jpeg_parser {

enum OutputColorspace { Original, RGB, Grayscale };
enum ParseStatus { Success, ResizeRequired, Failed };

unsigned char * parse_file(
  const std::string & jpeg_filename,
  int & width_out,
  int & height_out,
  int & bytes_per_pixel_out,
  OutputColorspace output_colorspace = Original,
  const int downsample_to_minimum_dimension = 0,
  float * applied_scale_factor_out = NULL);

unsigned char * parse_data(
  const unsigned char * const jpeg_data,
  const int jpeg_data_size,
  int & width_out,
  int & height_out,
  int & bytes_per_pixel_out,
  OutputColorspace output_colorspace = Original,
  const int downsample_to_minimum_dimension = 0,
  float * applied_scale_factor_out = NULL);

ParseStatus parse_file_preallocated_image_data(
  const std::string & jpeg_filename,
  unsigned char * const preallocated_image_data,
  const int preallocated_image_data_size,
  int & width_out,
  int & height_out,
  int & bytes_per_pixel_out,
  OutputColorspace output_colorspace = Original,
  const int downsample_to_minimum_dimension = 0,
  float * applied_scale_factor_out = NULL);

ParseStatus parse_data_preallocated_image_data(
  const unsigned char * const jpeg_data,
  const int jpeg_data_size,
  unsigned char * const preallocated_image_data,
  const int preallocated_image_data_size,
  int & width_out,
  int & height_out,
  int & bytes_per_pixel_out,
  OutputColorspace output_colorspace = Original,
  const int downsample_to_minimum_dimension = 0,
  float * applied_scale_factor_out = NULL);

} // namespace jpeg_parser

namespace core {

namespace colorspace {

inline jpeg_parser::OutputColorspace convert_to_jpeg_parser(const Colorspace colorspace)
{
  switch (colorspace)
  {
    case RGB:
      return jpeg_parser::RGB;
    case Grayscale:
      return jpeg_parser::Grayscale;
  }
  std::cerr << "ERROR: unrecognized colorspace" << std::endl;
  return jpeg_parser::RGB;
}

} // namespace colorspace;

} // namespace core

#endif // JPEG_PARSER_H

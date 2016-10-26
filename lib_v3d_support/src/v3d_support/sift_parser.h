#pragma once
#ifndef SIFT_PARSER_H
#define SIFT_PARSER_H

#include <features/sift_keypoint.h>
#include <features/sift_descriptor.h>
#include <string>
#include <vector>

namespace v3d_support {

namespace sift_parser {

static const std::string sift_header = "SIFT_COMPACT";
static const std::string sift_footer = "EOF";

int read_from_file(
  const std::string & sift_file_path,
  std::vector<features::SiftKeypoint> & returned_keypoints,
  std::vector<features::SiftDescriptorUChar> & returned_descriptors,
  const int max_num_features = -1,
  bool skip_descriptors_and_eof = false);

int read_from_memory(
  const unsigned char * const data,
  const int data_size,
  std::vector<features::SiftKeypoint> & returned_keypoints,
  std::vector<features::SiftDescriptorUChar> & returned_descriptors,
  const int max_num_features = -1);

void write_to_file(
  const std::string & sift_file_path,
  const std::vector<features::SiftKeypoint> & keypoints,
  const std::vector<features::SiftDescriptorUChar> & descriptors);

void write_to_memory(
  std::vector<unsigned char> & data,
  const std::vector<features::SiftKeypoint> & keypoints,
  const std::vector<features::SiftDescriptorUChar> & descriptors);

} // namespace sift_parser

} // namespace v3d_support

#endif // SIFT_PARSER_H

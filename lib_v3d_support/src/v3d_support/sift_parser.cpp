#include <v3d_support/sift_parser.h>
#include <core/file_helper.h>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>

int v3d_support::sift_parser::read_from_file(
  const std::string & file_path,
  std::vector<features::SiftKeypoint> & returned_keypoints,
  std::vector<features::SiftDescriptorUChar> & returned_descriptors,
  const int max_num_features,
  bool skip_descriptors_and_eof)
{
  returned_keypoints.clear();
  returned_descriptors.clear();

  FILE * file = core::file_helper::open_file(file_path, "rb");

  // Read the SIFT header.
  char buffer[16];
  size_t num_read = fread(buffer, 1, sift_header.length(), file);
  if (num_read != sift_header.length())
  {
    std::cerr << "ERROR: sift_parser::read_from_file() - unexpected end of file," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    return 0;
  }
  buffer[sift_header.length()] = 0;
  if (sift_header != buffer)
  {
    std::cerr << "ERROR: sift_parser::read_from_file() - missing expected header, "
      << sift_header << "," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    return 0;
  }

  // Read the number of features in the file.
  int num_features_in_file = 0;
  num_read = fread(&num_features_in_file, sizeof(int), 1, file);
  if (static_cast<int>(num_read) != 1)
  {
    std::cerr << "ERROR: sift_parser::read_from_file() - unexpected end of file," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    return 0;
  }

  const int num_features_to_read =
    (num_features_in_file > max_num_features && max_num_features > 0) ?
      max_num_features : num_features_in_file;

  returned_keypoints.resize(num_features_to_read);
  if (!skip_descriptors_and_eof)
  {
    returned_descriptors.resize(num_features_to_read);
  }

  // Read the SIFT keypoints.
  num_read = fread(&returned_keypoints[0], sizeof(features::SiftKeypoint), num_features_to_read, file);
  if (static_cast<int>(num_read) != num_features_to_read)
  {
    std::cerr << "ERROR: sift_parser::read_from_file() - unexpected end of file," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    returned_keypoints.clear();
    returned_descriptors.clear();
    return 0;
  }

  if (skip_descriptors_and_eof)
  {
    fclose(file);
    return num_features_to_read;
  }

  const int num_features_to_skip = num_features_in_file - num_features_to_read;
  if (num_features_to_skip > 0)
  {
    const int error_flag = fseek(file, num_features_to_skip * sizeof(features::SiftKeypoint), SEEK_CUR);
    if (error_flag != 0)
    {
      std::cerr << "ERROR: sift_parser::read_from_file() - unexpected end of file," << std::endl;
      std::cerr << file_path << std::endl;
      fclose(file);
      returned_keypoints.clear();
      returned_descriptors.clear();
      return 0;
    }
  }

  // Read the SIFT descriptors.
  num_read = fread(&returned_descriptors[0], sizeof(features::SiftDescriptorUChar), num_features_to_read, file);
  if (static_cast<int>(num_read) != num_features_to_read)
  {
    std::cerr << "ERROR: sift_parser::read_from_file() - unexpected end of file," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    returned_keypoints.clear();
    returned_descriptors.clear();
    return 0;
  }

  if (num_features_to_skip > 0)
  {
    const int error_flag = fseek(file, num_features_to_skip * sizeof(features::SiftDescriptorUChar), SEEK_CUR);
    if (error_flag != 0)
    {
      std::cerr << "ERROR: sift_parser::read_from_file() - unexpected end of file," << std::endl;
      std::cerr << file_path << std::endl;
      fclose(file);
      returned_keypoints.clear();
      returned_descriptors.clear();
      return 0;
    }
  }

  // Read the SIFT footer.
  num_read = fread(buffer, 1, sift_footer.length(), file);
  if (num_read != sift_footer.length())
  {
    std::cerr << "ERROR: sift_parser::read_from_file() - unexpected end of file," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    returned_keypoints.clear();
    returned_descriptors.clear();
    return 0;
  }
  buffer[sift_footer.length()] = 0;
  if (sift_footer != buffer)
  {
    std::cerr << "ERROR: sift_parser::read_from_file() - missing expected footer, "
      << sift_footer << "," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    returned_keypoints.clear();
    returned_descriptors.clear();
    return 0;
  }
  
  fclose(file);
  return num_features_to_read;
}

int v3d_support::sift_parser::read_from_memory(
  const unsigned char * const data,
  const int data_size,
  std::vector<features::SiftKeypoint> & returned_keypoints,
  std::vector<features::SiftDescriptorUChar> & returned_descriptors,
  const int max_num_features)
{
  returned_keypoints.clear();
  returned_descriptors.clear();

  const unsigned char * data_ptr = data;
  int remaining_data_size = data_size;

  // Read the SIFT header.
  if (remaining_data_size < sift_header.length())
  {
    std::cerr << "ERROR: sift_parser::read_from_memory() - unexpected end of data" << std::endl;
    return 0;
  }
  if (strncmp(sift_header.c_str(), (const char *)data_ptr, sift_header.length()) != 0)
  {
    std::cerr << "ERROR: sift_parser::read_from_memory() - missing expected header, "
      << sift_header << "," << std::endl;
    return 0;
  }
  data_ptr += sift_header.length();
  remaining_data_size -= static_cast<int>(sift_header.length());

  // Read the number of features.
  if (remaining_data_size < static_cast<int>(sizeof(unsigned int)))
  {
    std::cerr << "ERROR: sift_parser::read_from_memory() - unexpected end of data" << std::endl;
    return 0;
  }
  int num_features_in_data = *(const int *)data_ptr;
  data_ptr += sizeof(int);
  remaining_data_size -= sizeof(int);

  const int num_features_to_read =
    (num_features_in_data > max_num_features && max_num_features > 0) ?
      max_num_features : num_features_in_data;

  returned_keypoints.resize(num_features_to_read);
  returned_descriptors.resize(num_features_to_read);

  // Read the SIFT keypoints.
  if (remaining_data_size < num_features_to_read * static_cast<int>(sizeof(features::SiftKeypoint)))
  {
    std::cerr << "ERROR: sift_parser::read_from_memory() - unexpected end of data" << std::endl;
    returned_keypoints.clear();
    returned_descriptors.clear();
    return 0;
  }
#ifdef WIN32
  memcpy_s(
    &returned_keypoints[0], // destination
    num_features_to_read * sizeof(features::SiftKeypoint), // destination size
    data_ptr, // source
    num_features_to_read * sizeof(features::SiftKeypoint)); // num bytes
#else
  memcpy(
    &returned_keypoints[0], // destination
    data_ptr, // source
    num_features_to_read * sizeof(features::SiftKeypoint)); // num bytes
#endif
  data_ptr += num_features_to_read * sizeof(features::SiftKeypoint);
  remaining_data_size -= num_features_to_read * sizeof(features::SiftKeypoint);

  const int num_features_to_skip = num_features_in_data - num_features_to_read;
  if (num_features_to_skip > 0)
  {
    if (remaining_data_size < num_features_to_skip * static_cast<int>(sizeof(features::SiftKeypoint)))
    {
      std::cerr << "ERROR: sift_parser::read_from_memory() - unexpected end of data" << std::endl;
      returned_keypoints.clear();
      returned_descriptors.clear();
      return 0;
    }
    data_ptr += num_features_to_skip * sizeof(features::SiftKeypoint);
    remaining_data_size -= num_features_to_skip * sizeof(features::SiftKeypoint);
  }

  // Read the SIFT descriptors.
  if (remaining_data_size < num_features_to_read * static_cast<int>(sizeof(features::SiftDescriptorUChar)))
  {
    std::cerr << "ERROR: sift_parser::read_from_memory() - unexpected end of data" << std::endl;
    returned_keypoints.clear();
    returned_descriptors.clear();
    return 0;
  }
#ifdef WIN32
  memcpy_s(
    &returned_descriptors[0], // destination
    num_features_to_read * sizeof(features::SiftDescriptorUChar), // destination size
    data_ptr, // source
    num_features_to_read * sizeof(features::SiftDescriptorUChar)); // num bytes
#else
  memcpy(
    &returned_descriptors[0], // destination
    data_ptr, // source
    num_features_to_read * sizeof(features::SiftDescriptorUChar)); // num bytes
#endif
  data_ptr += num_features_to_read * sizeof(features::SiftDescriptorUChar);
  remaining_data_size -= num_features_to_read * sizeof(features::SiftDescriptorUChar);

  if (num_features_to_skip > 0)
  {
    if (remaining_data_size < num_features_to_skip * static_cast<int>(sizeof(features::SiftDescriptorUChar)))
    {
      std::cerr << "ERROR: sift_parser::read_from_memory() - unexpected end of data" << std::endl;
      returned_keypoints.clear();
      returned_descriptors.clear();
      return 0;
    }
    data_ptr += num_features_to_skip * sizeof(features::SiftDescriptorUChar);
    remaining_data_size -= num_features_to_skip * sizeof(features::SiftDescriptorUChar);
  }
  
  // Read the SIFT footer.
  if (remaining_data_size < sift_footer.length())
  {
    std::cerr << "ERROR: sift_parser::read_from_memory() - unexpected end of data" << std::endl;
    returned_keypoints.clear();
    returned_descriptors.clear();
    return 0;
  }
  if (strncmp(sift_footer.c_str(), (const char *)data_ptr, sift_footer.length()) != 0)
  {
    std::cerr << "ERROR: sift_parser::read_from_memory() - missing expected footer, "
      << sift_footer << "," << std::endl;
    returned_keypoints.clear();
    returned_descriptors.clear();
    return 0;
  }
  remaining_data_size -= static_cast<int>(sift_footer.length());

  if (remaining_data_size != 0)
  {
    std::cerr << "ERROR: sift_parser::read_from_memory() - extra data at end" << std::endl;
    returned_keypoints.clear();
    returned_descriptors.clear();
    return 0;
  }

  return num_features_to_read;
}

void v3d_support::sift_parser::write_to_file(
  const std::string & file_path,
  const std::vector<features::SiftKeypoint> & keypoints,
  const std::vector<features::SiftDescriptorUChar> & descriptors)
{
  if (keypoints.size() != descriptors.size())
  {
    std::cerr << "ERROR: sift_parser::write_to_file() - different number of keypoints and descriptors" << std::endl;
    exit(EXIT_FAILURE);
  }

  FILE * file = core::file_helper::open_file(file_path, "wb");

  // Write the SIFT header.
  size_t num_written = fwrite(sift_header.c_str(), sizeof(char), sift_header.length(), file);
  if (num_written != sift_header.length())
  {
    std::cerr << "ERROR: sift_parser::write_to_file() - unable to write all data," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    exit(EXIT_FAILURE);
  }

  // Write the number of features.
  const int num_features = static_cast<int>(keypoints.size());
  num_written = fwrite(&num_features, sizeof(int), 1, file);
  if (num_written != 1)
  {
    std::cerr << "ERROR: sift_parser::write_to_file() - unable to write all data," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    exit(EXIT_FAILURE);
  }

  // Write the SIFT keypoints.
  num_written = fwrite(&keypoints[0], sizeof(features::SiftKeypoint), num_features, file);
  if (static_cast<int>(num_written) != num_features)
  {
    std::cerr << "ERROR: sift_parser::write_to_file() - unable to write all data," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    exit(EXIT_FAILURE);
  }

  // Write the SIFT descriptors.
  num_written = fwrite(&descriptors[0], sizeof(features::SiftDescriptorUChar), num_features, file);
  if (static_cast<int>(num_written) != num_features)
  {
    std::cerr << "ERROR: sift_parser::write_to_file() - unable to write all data," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    exit(EXIT_FAILURE);
  }

  // Write the SIFT footer.
  num_written = fwrite(sift_footer.c_str(), sizeof(char), sift_footer.length(), file);
  if (num_written != sift_footer.length())
  {
    std::cerr << "ERROR: sift_parser::write_to_file() - unable to write all data," << std::endl;
    std::cerr << file_path << std::endl;
    fclose(file);
    exit(EXIT_FAILURE);
  }

  fclose(file);
}

void v3d_support::sift_parser::write_to_memory(
  std::vector<unsigned char> & data,
  const std::vector<features::SiftKeypoint> & keypoints,
  const std::vector<features::SiftDescriptorUChar> & descriptors)
{
  if (keypoints.size() != descriptors.size())
  {
    std::cerr << "ERROR: sift_parser::write_to_memory() - different number of keypoints and descriptors" << std::endl;
    exit(EXIT_FAILURE);
  }

  const int num_features = static_cast<int>(keypoints.size());

  const int data_size =
    static_cast<int>(sift_header.length()) + // header
    sizeof(int) + // num features
    num_features * sizeof(features::SiftKeypoint) + // keypoints
    num_features * sizeof(features::SiftDescriptorUChar) + // descriptors
    static_cast<int>(sift_footer.length()); // footer

  data.resize(data_size);
  int current_data_offset = 0;

  // Write SIFT header.
#ifdef WIN32
  memcpy_s(
    &data[current_data_offset], // destination
    data_size - current_data_offset, // destination size
    sift_header.c_str(), // source
    sift_header.length()); // num bytes
#else
  memcpy(
    &data[current_data_offset], // destination
    sift_header.c_str(), // source
    sift_header.length()); // num bytes
#endif
  current_data_offset += static_cast<int>(sift_header.length());

  // Write num features.
#ifdef WIN32
  memcpy_s(
    &data[current_data_offset], // destination
    data_size - current_data_offset, // destination size
    &num_features, // source
    sizeof(int)); // num bytes
#else
  memcpy(
    &data[current_data_offset], // destination
    &num_features, // source
    sizeof(int)); // num bytes
#endif
  current_data_offset += sizeof(int);

  // Write SIFT keypoints.
#ifdef WIN32
  memcpy_s(
    &data[current_data_offset], // destination
    data_size - current_data_offset, // destination size
    &keypoints[0], // source
    num_features * sizeof(features::SiftKeypoint)); // num bytes
#else
  memcpy(
    &data[current_data_offset], // destination
    &keypoints[0], // source
    num_features * sizeof(features::SiftKeypoint)); // num bytes
#endif
  current_data_offset += num_features * sizeof(features::SiftKeypoint);

  // Write SIFT descriptors.
#ifdef WIN32
  memcpy_s(
    &data[current_data_offset], // destination
    data_size - current_data_offset, // destination size
    &descriptors[0], // source
    num_features * sizeof(features::SiftDescriptorUChar)); // num bytes
#else
  memcpy(
    &data[current_data_offset], // destination
    &descriptors[0], // source
    num_features * sizeof(features::SiftDescriptorUChar)); // num bytes
#endif
  current_data_offset += num_features * sizeof(features::SiftDescriptorUChar);

  // Write SFIT footer.
#ifdef WIN32
  memcpy_s(
    &data[current_data_offset], // destination
    data_size - current_data_offset, // destination size
    sift_footer.c_str(), // source
    sift_footer.length()); // num bytes
#else
  memcpy(
    &data[current_data_offset], // destination
    sift_footer.c_str(), // source
    sift_footer.length()); // num bytes
#endif
  current_data_offset += static_cast<int>(sift_footer.length());

  if (current_data_offset != data_size)
  {
    std::cerr << "ERROR: sift_parser::write_to_memory() - unexpected data size" << std::endl;
    exit(EXIT_FAILURE);
  }
}

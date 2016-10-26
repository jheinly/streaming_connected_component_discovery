#include <dataset/focal_lookup.h>
#include <core/file_helper.h>
#include <iostream>

dataset::FocalLookup::FocalLookup()
: m_image_name_to_dimensions_map(),
  m_image_name_to_focal_map()
{}

dataset::FocalLookup::~FocalLookup()
{}

dataset::FocalLookup::Result dataset::FocalLookup::lookup_focal(
  const std::string & image_name,
  const core::ImageDimensions & image_dimensions,
  const float image_scale_factor) const
{
  // Attempt to find image dimensions and focal entries for this image.
  boost::unordered_map<std::string, core::ImageDimensions>::const_iterator found_dimensions =
    m_image_name_to_dimensions_map.find(image_name);
  boost::unordered_map<std::string, float>::const_iterator found_focal =
    m_image_name_to_focal_map.find(image_name);

  if (found_dimensions != m_image_name_to_dimensions_map.end() &&
      found_focal != m_image_name_to_focal_map.end())
  {
    // If both the dimensions and focal for this image were found,
    // compute the scale between the dataset's dimensions and the
    // provided dimensions to compute the new focal length.

    const float scale =
      static_cast<float>(image_dimensions.max_dimension()) /
      static_cast<float>(found_dimensions->second.max_dimension());

    return verify_focal_and_create_result(
      image_dimensions,
      scale * found_focal->second,
      true);
  }
  else if (found_focal != m_image_name_to_focal_map.end())
  {
    // If the dataset contained a focal length entry for this image
    // but not image dimensions, attempt to use the provided scale
    // factor to compute a new focal. However, don't trust the focal
    // as being provided by EXIF data.

    if (image_scale_factor > 0)
    {
      return verify_focal_and_create_result(
        image_dimensions,
        image_scale_factor * found_focal->second,
        false);
    }
    else
    {
      return compute_default_focal_and_create_result(image_dimensions);
    }
  }
  else
  {
    // No dataset information was found for this image, so compute
    // a default focal length.

    return compute_default_focal_and_create_result(image_dimensions);
  }
}

void dataset::FocalLookup::load_dimension_list(const std::string & dimension_list_filepath)
{
  m_image_name_to_dimensions_map.clear();

  FILE * file = core::file_helper::open_file(dimension_list_filepath, "r");

  const int buffer_size = 256;
  char buffer[buffer_size];

  const int format_size = 32;
  char format[format_size];
#ifdef WIN32
  sprintf_s(format, format_size, "%%%ds %%d %%d\n", buffer_size - 1);
#else
  sprintf(format, "%%%ds %%d %%d\n", buffer_size - 1);
#endif

  std::string image_name;
  int width;
  int height;

  for (;;)
  {
#ifdef WIN32
    const int num_read = fscanf_s(file, format, buffer, buffer_size, &width, &height);
#else
    const int num_read = fscanf(file, format, buffer, &width, &height);
#endif
    if (num_read != 3)
    {
      break;
    }

    image_name = buffer;

    boost::unordered_map<std::string, core::ImageDimensions>::const_iterator found_entry =
      m_image_name_to_dimensions_map.find(image_name);

    if (found_entry == m_image_name_to_dimensions_map.end())
    {
      m_image_name_to_dimensions_map[image_name] = core::ImageDimensions(width, height);
    }
    else
    {
      std::cerr << "WARNING: duplicate entry found in dimension list for image, " <<
        image_name << std::endl;
    }
  }

  if (!feof(file))
  {
    std::cerr << "ERROR: format error within dimension file after reading " <<
      m_image_name_to_dimensions_map.size() << " entries" << std::endl;
    fclose(file);
    exit(EXIT_FAILURE);
  }

  fclose(file);
}

void dataset::FocalLookup::load_focal_list(const std::string & focal_list_filepath)
{
  m_image_name_to_focal_map.clear();

  FILE * file = core::file_helper::open_file(focal_list_filepath, "r");

  const int buffer_size = 256;
  char buffer[buffer_size];

  const int format_size = 32;
  char format[format_size];
#ifdef WIN32
  sprintf_s(format, format_size, "%%%ds %%f\n", buffer_size - 1);
#else
  sprintf(format, "%%%ds %%f\n", buffer_size - 1);
#endif

  std::string image_name;
  float focal;

  for (;;)
  {
#ifdef WIN32
    const int num_read = fscanf_s(file, format, buffer, buffer_size, &focal);
#else
    const int num_read = fscanf(file, format, buffer, &focal);
#endif
    if (num_read != 2)
    {
      break;
    }

    image_name = buffer;

    boost::unordered_map<std::string, float>::const_iterator found_entry =
      m_image_name_to_focal_map.find(image_name);

    if (found_entry == m_image_name_to_focal_map.end())
    {
      m_image_name_to_focal_map[image_name] = focal;
    }
    else
    {
      std::cerr << "WARNING: duplicate entry found in focal list for image, " <<
        image_name << std::endl;
    }
  }

  if (!feof(file))
  {
    std::cerr << "ERROR: format error within focal file after reading " <<
      m_image_name_to_focal_map.size() << " entries" << std::endl;
    fclose(file);
    exit(EXIT_FAILURE);
  }

  fclose(file);
}

void dataset::FocalLookup::load_dimension_list_with_thread(
  const std::string & dimension_list_filepath)
{
  m_dimension_list_thread = boost::thread(
    &FocalLookup::load_dimension_list, this, dimension_list_filepath);
}

void dataset::FocalLookup::load_focal_list_with_thread(
  const std::string & focal_list_filepath)
{
  m_focal_list_thread = boost::thread(
    &FocalLookup::load_focal_list, this, focal_list_filepath);
}

void dataset::FocalLookup::wait_for_thread_to_load_dimension_list()
{
  m_dimension_list_thread.join();
}

void dataset::FocalLookup::wait_for_thread_to_load_focal_list()
{
  m_focal_list_thread.join();
}

dataset::FocalLookup::Result dataset::FocalLookup::compute_default_focal_and_create_result(
  const core::ImageDimensions & image_dimensions) const
{
  // For images with a missing or invalid focal length, set the focal/dimension ratio
  // to a "normal" amount, which is a 43.3mm lens on a full frame camera.
  // http://en.wikipedia.org/wiki/Normal_lens
  const float default_focal_width_ratio = 43.3f / 36.0f;

  return Result(
    default_focal_width_ratio * static_cast<float>(image_dimensions.max_dimension()),
    false);
}

dataset::FocalLookup::Result dataset::FocalLookup::verify_focal_and_create_result(
  const core::ImageDimensions & image_dimensions,
  const float focal,
  const bool is_focal_from_exif) const
{
  // Enforce a maximum ratio between the focal length and dimension of an image.
  // Here, set the maximum to be equivalent to a 1000mm lens on a full frame camera.
  const float max_focal_dimension_ratio = 1000.0f / 36.0f;

  // Enforce a minimum ratio between the focal length and dimension of an image.
  // Here, set the minimum to be equivalent to a 10mm lens on a full frame camera.
  const float min_focal_dimension_ratio = 10.0f / 36.0f;

  const float ratio = focal / static_cast<float>(image_dimensions.max_dimension());

  if (ratio < min_focal_dimension_ratio || ratio > max_focal_dimension_ratio)
  {
    return compute_default_focal_and_create_result(image_dimensions);
  }
  else
  {
    return Result(focal, is_focal_from_exif);
  }
}

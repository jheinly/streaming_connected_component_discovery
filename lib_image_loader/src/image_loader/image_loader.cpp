#include <image_loader/image_loader.h>
#include <core/colorspace_opencv.h>

image_loader::ImageLoader::ImageLoader(
  const core::colorspace::Colorspace output_image_colorspace,
  const image_resizer::ImageResizer::ResizeMode output_image_resize_mode,
  const int resized_image_dimension,
  const int minimum_allowed_image_dimension,
  const float maximum_allowed_aspect_ratio,
  const int maximum_loaded_image_dimension_hint)
: m_output_image_colorspace(output_image_colorspace),
  m_image_resizer(output_image_resize_mode, resized_image_dimension),
  m_minimum_allowed_image_dimension(minimum_allowed_image_dimension),
  m_maximum_allowed_aspect_ratio(maximum_allowed_aspect_ratio),
  m_loaded_image_data(NULL),
  m_loaded_image_data_size(0),
  m_loaded_image_width(0),
  m_loaded_image_height(0),
  m_applied_scale_factor(0.0f)
{
  if (maximum_loaded_image_dimension_hint > 0)
  {
    m_loaded_image_data_size =
      maximum_loaded_image_dimension_hint * maximum_loaded_image_dimension_hint *
      core::colorspace::num_bytes_per_pixel(output_image_colorspace);
  }
  else
  {
    // A dummy value that will be replaced the first time an image is loaded.
    m_loaded_image_data_size = core::colorspace::num_bytes_per_pixel(output_image_colorspace);
  }

  m_loaded_image_data = new unsigned char[m_loaded_image_data_size];

  if (m_loaded_image_data == NULL)
  {
    std::cerr << "ERROR: ImageLoader - failed to allocate new image data" << std::endl;
    m_loaded_image_data_size = 0;
    return;
  }
}

image_loader::ImageLoader::~ImageLoader()
{
  if (m_loaded_image_data != NULL)
  {
    delete[] m_loaded_image_data;
    m_loaded_image_data = NULL;
  }
}

bool image_loader::ImageLoader::start_loading_jpeg_from_file(
  const std::string & jpeg_filename)
{
  return start_loading_helper(LoadFromFile(jpeg_filename));
}

bool image_loader::ImageLoader::start_loading_jpeg_from_data(
  const unsigned char * const jpeg_data,
  const int jpeg_data_size)
{
  return start_loading_helper(LoadFromData(jpeg_data, jpeg_data_size));
}

image_resizer::ImageResizer::ResizeResult image_loader::ImageLoader::finish_loading_jpeg(
  unsigned char * const preallocated_image_data,
  const int preallocated_image_data_size)
{
  image_resizer::ImageResizer::ResizeResult resize_result = m_image_resizer.resize_image(
    m_loaded_image_data,
    m_loaded_image_width,
    m_loaded_image_height,
    m_output_image_colorspace,
    preallocated_image_data,
    preallocated_image_data_size);
  resize_result.scale *= m_applied_scale_factor;
  return resize_result;
}

jpeg_parser::ParseStatus image_loader::ImageLoader::LoadFromFile::operator()(
  unsigned char * const preallocated_image_data,
  const int preallocated_image_data_size,
  int & width_out,
  int & height_out,
  int & bytes_per_pixel_out,
  core::colorspace::Colorspace output_colorspace,
  const int downsample_to_minimum_dimension,
  float * applied_scale_factor_out) const
{
  return jpeg_parser::parse_file_preallocated_image_data(
    m_jpeg_filename,
    preallocated_image_data,
    preallocated_image_data_size,
    width_out,
    height_out,
    bytes_per_pixel_out,
    core::colorspace::convert_to_jpeg_parser(output_colorspace),
    downsample_to_minimum_dimension,
    applied_scale_factor_out);
}

jpeg_parser::ParseStatus image_loader::ImageLoader::LoadFromData::operator()(
  unsigned char * const preallocated_image_data,
  const int preallocated_image_data_size,
  int & width_out,
  int & height_out,
  int & bytes_per_pixel_out,
  core::colorspace::Colorspace output_colorspace,
  const int downsample_to_minimum_dimension,
  float * applied_scale_factor_out) const
{
  return jpeg_parser::parse_data_preallocated_image_data(
    m_jpeg_data,
    m_jpeg_data_size,
    preallocated_image_data,
    preallocated_image_data_size,
    width_out,
    height_out,
    bytes_per_pixel_out,
    core::colorspace::convert_to_jpeg_parser(output_colorspace),
    downsample_to_minimum_dimension,
    applied_scale_factor_out);
}

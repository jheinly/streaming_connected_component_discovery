#include <jpeg_parser/jpeg_parser.h>
#include <core/file_helper.h>
#include <cstdio>
#include <iostream>
#include <libjpeg_turbo/jpeglib.h>

namespace jpeg_parser {

struct my_error_mgr
{
  jpeg_error_mgr jpeg;
  jmp_buf setjmp_buffer;
};

static void my_error_exit(j_common_ptr cinfo)
{
  // cinfo->err actually points to a my_error_mgr object
  my_error_mgr * error_mgr = (my_error_mgr *)cinfo->err;

  char buffer[JMSG_LENGTH_MAX];

  // Get the error message
  (*cinfo->err->format_message)(cinfo, buffer);

  // Display the error
  std::cerr << "JpegParser ERROR: " << buffer << std::endl;

  // Return control to the setjmp point
  longjmp(error_mgr->setjmp_buffer, 1);
}

} // namespace jpeg_parser

unsigned char * jpeg_parser::parse_file(
  const std::string & jpeg_filename,
  int & width_out,
  int & height_out,
  int & bytes_per_pixel_out,
  OutputColorspace output_colorspace,
  const int downsample_to_minimum_dimension,
  float * applied_scale_factor_out)
{
  jpeg_decompress_struct cinfo;
  my_error_mgr error_mgr;

  width_out = 0;
  height_out = 0;
  bytes_per_pixel_out = 0;

  // Attempt to open the file for reading
  FILE * file = NULL;
#ifdef WIN32
  fopen_s(&file, jpeg_filename.c_str(), "rb");
#else
  file = fopen(jpeg_filename.c_str(), "rb");
#endif
  if (file == NULL)
  {
    std::cerr << "jpeg_parser ERROR: failed to open file," << std::endl;
    std::cerr << "  \"" << jpeg_filename << "\"" << std::endl;
    return NULL;
  }

  // Set the normal JPEG error routines, and then override the error_exit function
  cinfo.err = jpeg_std_error(&error_mgr.jpeg);
  error_mgr.jpeg.error_exit = my_error_exit;

  // Set up the setjmp return context to use if the JPEG library encounters an error
  if (setjmp(error_mgr.setjmp_buffer))
  {
    // The JPEG library has encountered an error
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
    return NULL;
  }

  // Initialize the JPEG decompression object
  jpeg_create_decompress(&cinfo);

  // This seems to have very little impact on the speed
  cinfo.dct_method = JDCT_FASTEST;

  // Set the input data source
  jpeg_stdio_src(&cinfo, file);

  // Read file parameters
  jpeg_read_header(&cinfo, TRUE);

  // Set the output colorspace
  switch (output_colorspace)
  {
    case Original:
      break;
    case RGB:
      cinfo.out_color_space = JCS_RGB;
      break;
    case Grayscale:
      cinfo.out_color_space = JCS_GRAYSCALE;
      break;
  }

  // Check to see if a downsample_to_minimum_dimension value has been provided.
  if (downsample_to_minimum_dimension > 0)
  {
    // Compute the scale factors required to make the JPEG conform to the requested
    // image dimension.
    const float width_scale = float(cinfo.image_width) / downsample_to_minimum_dimension;
    const float height_scale = float(cinfo.image_height) / downsample_to_minimum_dimension;

    // Determine which scale is smaller, as that is the dimension that limits how much
    // the image can be resized while still maintaining the minimum requested image
    // dimension of downsample_to_minimum_dimension.
    const float min_scale = width_scale < height_scale ? width_scale : height_scale;

    if (min_scale < 2.0f)
    {
      // No-op, the image can't be downsized without making it smaller than the
      // requested image dimension of downsample_to_minimum_dimension.
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f;
      }
    }
    else if (min_scale < 4.0f)
    {
      // The image needs to be downsampled by a factor less than 4, so downsample
      // by the next-best factor, which is 2.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 2;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 2.0f;
      }
    }
    else if (min_scale < 8.0f)
    {
      // The image needs to be downsampled by a factor less than 8, so downsample
      // by the next-best factor, which is 4.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 4;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 4.0f;
      }
    }
    else
    {
      // The image needs to be downsampled by a factor of 8 or more, so downsample
      // by a factor of 8.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 8;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 8.0f;
      }
    }
  }
  else if (applied_scale_factor_out != NULL)
  {
    *applied_scale_factor_out = 1.0f;
  }

  // Start decompression
  jpeg_start_decompress(&cinfo);

  // Allocate image data
  const unsigned int image_data_size = cinfo.output_width * cinfo.output_height * cinfo.output_components;
  unsigned char * image_data = new unsigned char[image_data_size];
  if (image_data == NULL)
  {
    std::cerr << "jpeg_parser ERROR: out of memory, requested " << image_data_size << " bytes" << std::endl;
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
    return NULL;
  }

  width_out = cinfo.output_width;
  height_out = cinfo.output_height;
  bytes_per_pixel_out = cinfo.output_components;

  // Read the JPEG image, one row at a time
  const unsigned int row_stride = cinfo.output_width * cinfo.output_components;
  unsigned char * image_data_ptr = image_data;
  while (cinfo.output_scanline < cinfo.output_height)
  {
    // jpeg_read_scanlines expects an array of pointers to scanlines, so it
    // is easiest to just provide a pointer to the current row and read
    // one row at a time
    jpeg_read_scanlines(&cinfo, (JSAMPARRAY)(&image_data_ptr), 1);
    image_data_ptr += row_stride;
  }

  // Finish decompression
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(file);

  return image_data;
}

unsigned char * jpeg_parser::parse_data(
  const unsigned char * const jpeg_data,
  const int jpeg_data_size,
  int & width_out,
  int & height_out,
  int & bytes_per_pixel_out,
  OutputColorspace output_colorspace,
  const int downsample_to_minimum_dimension,
  float * applied_scale_factor_out)
{
  jpeg_decompress_struct cinfo;
  my_error_mgr error_mgr;

  width_out = 0;
  height_out = 0;
  bytes_per_pixel_out = 0;

  // Set the normal JPEG error routines, and then override the error_exit function
  cinfo.err = jpeg_std_error(&error_mgr.jpeg);
  error_mgr.jpeg.error_exit = my_error_exit;

  // Set up the setjmp return context to use if the JPEG library encounters an error
  if (setjmp(error_mgr.setjmp_buffer))
  {
    // The JPEG library has encountered an error
    jpeg_destroy_decompress(&cinfo);
    return NULL;
  }

  // Initialize the JPEG decompression object
  jpeg_create_decompress(&cinfo);

  // This seems to have very little impact on the speed
  cinfo.dct_method = JDCT_FASTEST;

  // Set the input data source
  jpeg_mem_src(&cinfo, (unsigned char *)jpeg_data, jpeg_data_size);

  // Read file parameters
  jpeg_read_header(&cinfo, TRUE);

  // Set the output colorspace
  switch (output_colorspace)
  {
    case Original:
      break;
    case RGB:
      cinfo.out_color_space = JCS_RGB;
      break;
    case Grayscale:
      cinfo.out_color_space = JCS_GRAYSCALE;
      break;
  }

  // Check to see if a downsample_to_minimum_dimension value has been provided.
  if (downsample_to_minimum_dimension > 0)
  {
    // Compute the scale factors required to make the JPEG conform to the requested
    // image dimension.
    const float width_scale = float(cinfo.image_width) / downsample_to_minimum_dimension;
    const float height_scale = float(cinfo.image_height) / downsample_to_minimum_dimension;

    // Determine which scale is smaller, as that is the dimension that limits how much
    // the image can be resized while still maintaining the minimum requested image
    // dimension of downsample_to_minimum_dimension.
    const float min_scale = width_scale < height_scale ? width_scale : height_scale;

    if (min_scale < 2.0f)
    {
      // No-op, the image can't be downsized without making it smaller than the
      // requested image dimension of downsample_to_minimum_dimension.
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f;
      }
    }
    else if (min_scale < 4.0f)
    {
      // The image needs to be downsampled by a factor less than 4, so downsample
      // by the next-best factor, which is 2.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 2;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 2.0f;
      }
    }
    else if (min_scale < 8.0f)
    {
      // The image needs to be downsampled by a factor less than 8, so downsample
      // by the next-best factor, which is 4.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 4;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 4.0f;
      }
    }
    else
    {
      // The image needs to be downsampled by a factor of 8 or more, so downsample
      // by a factor of 8.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 8;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 8.0f;
      }
    }
  }
  else if (applied_scale_factor_out != NULL)
  {
    *applied_scale_factor_out = 1.0f;
  }

  // Start decompression
  jpeg_start_decompress(&cinfo);

  // Allocate image data
  const unsigned int image_data_size = cinfo.output_width * cinfo.output_height * cinfo.output_components;
  unsigned char * image_data = new unsigned char[image_data_size];
  if (image_data == NULL)
  {
    std::cerr << "jpeg_parser ERROR: out of memory, requested " << image_data_size << " bytes" << std::endl;
    jpeg_destroy_decompress(&cinfo);
    return NULL;
  }

  width_out = cinfo.output_width;
  height_out = cinfo.output_height;
  bytes_per_pixel_out = cinfo.output_components;

  // Read the JPEG image, one row at a time
  const unsigned int row_stride = cinfo.output_width * cinfo.output_components;
  unsigned char * image_data_ptr = image_data;
  while (cinfo.output_scanline < cinfo.output_height)
  {
    // jpeg_read_scanlines expects an array of pointers to scanlines, so it
    // is easiest to just provide a pointer to the current row and read
    // one row at a time
    jpeg_read_scanlines(&cinfo, (JSAMPARRAY)(&image_data_ptr), 1);
    image_data_ptr += row_stride;
  }

  // Finish decompression
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  return image_data;
}

jpeg_parser::ParseStatus jpeg_parser::parse_file_preallocated_image_data(
  const std::string & jpeg_filename,
  unsigned char * const preallocated_image_data,
  const int preallocated_image_data_size,
  int & width_out,
  int & height_out,
  int & bytes_per_pixel_out,
  OutputColorspace output_colorspace,
  const int downsample_to_minimum_dimension,
  float * applied_scale_factor_out)
{
  jpeg_decompress_struct cinfo;
  my_error_mgr error_mgr;

  width_out = 0;
  height_out = 0;
  bytes_per_pixel_out = 0;

  // Attempt to open the file for reading
  FILE * file = NULL;
#ifdef WIN32
  fopen_s(&file, jpeg_filename.c_str(), "rb");
#else
  file = fopen(jpeg_filename.c_str(), "rb");
#endif
  if (file == NULL)
  {
    std::cerr << "jpeg_parser ERROR: failed to open file," << std::endl;
    std::cerr << "  \"" << jpeg_filename << "\"" << std::endl;
    return Failed;
  }

  // Set the normal JPEG error routines, and then override the error_exit function
  cinfo.err = jpeg_std_error(&error_mgr.jpeg);
  error_mgr.jpeg.error_exit = my_error_exit;

  // Set up the setjmp return context to use if the JPEG library encounters an error
  if (setjmp(error_mgr.setjmp_buffer))
  {
    // The JPEG library has encountered an error
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
    return Failed;
  }

  // Initialize the JPEG decompression object
  jpeg_create_decompress(&cinfo);

  // This seems to have very little impact on the speed
  cinfo.dct_method = JDCT_FASTEST;

  // Set the input data source
  jpeg_stdio_src(&cinfo, file);

  // Read file parameters
  jpeg_read_header(&cinfo, TRUE);

  // Set the output colorspace
  switch (output_colorspace)
  {
    case Original:
      break;
    case RGB:
      cinfo.out_color_space = JCS_RGB;
      break;
    case Grayscale:
      cinfo.out_color_space = JCS_GRAYSCALE;
      break;
  }

  // Check to see if a downsample_to_minimum_dimension value has been provided.
  if (downsample_to_minimum_dimension > 0)
  {
    // Compute the scale factors required to make the JPEG conform to the requested
    // image dimension.
    const float width_scale = float(cinfo.image_width) / downsample_to_minimum_dimension;
    const float height_scale = float(cinfo.image_height) / downsample_to_minimum_dimension;

    // Determine which scale is smaller, as that is the dimension that limits how much
    // the image can be resized while still maintaining the minimum requested image
    // dimension of downsample_to_minimum_dimension.
    const float min_scale = width_scale < height_scale ? width_scale : height_scale;

    if (min_scale < 2.0f)
    {
      // No-op, the image can't be downsized without making it smaller than the
      // requested image dimension of downsample_to_minimum_dimension.
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f;
      }
    }
    else if (min_scale < 4.0f)
    {
      // The image needs to be downsampled by a factor less than 4, so downsample
      // by the next-best factor, which is 2.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 2;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 2.0f;
      }
    }
    else if (min_scale < 8.0f)
    {
      // The image needs to be downsampled by a factor less than 8, so downsample
      // by the next-best factor, which is 4.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 4;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 4.0f;
      }
    }
    else
    {
      // The image needs to be downsampled by a factor of 8 or more, so downsample
      // by a factor of 8.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 8;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 8.0f;
      }
    }
  }
  else if (applied_scale_factor_out != NULL)
  {
    *applied_scale_factor_out = 1.0f;
  }

  // Start decompression
  jpeg_start_decompress(&cinfo);

  width_out = cinfo.output_width;
  height_out = cinfo.output_height;
  bytes_per_pixel_out = cinfo.output_components;

  // Check that there is sufficient preallocated data
  const int image_data_size = cinfo.output_width * cinfo.output_height * cinfo.output_components;
  if (image_data_size > preallocated_image_data_size)
  {
    // Don't print an error message to allow the caller to detect the failed
    // decode, resize their image buffer, and try again.
    //std::cerr << "jpeg_parser ERROR: not enough preallocated memory," << std::endl;
    //std::cerr << "  requested " << image_data_size << " bytes, but "
    //          << preallocated_image_data_size << " bytes available" << std::endl;
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
    return ResizeRequired;
  }

  // Read the JPEG image, one row at a time
  const unsigned int row_stride = cinfo.output_width * cinfo.output_components;
  unsigned char * image_data_ptr = preallocated_image_data;
  while (cinfo.output_scanline < cinfo.output_height)
  {
    // jpeg_read_scanlines expects an array of pointers to scanlines, so it
    // is easiest to just provide a pointer to the current row and read
    // one row at a time
    jpeg_read_scanlines(&cinfo, (JSAMPARRAY)(&image_data_ptr), 1);
    image_data_ptr += row_stride;
  }

  // Finish decompression
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(file);

  return Success;
}

jpeg_parser::ParseStatus jpeg_parser::parse_data_preallocated_image_data(
  const unsigned char * const jpeg_data,
  const int jpeg_data_size,
  unsigned char * const preallocated_image_data,
  const int preallocated_image_data_size,
  int & width_out,
  int & height_out,
  int & bytes_per_pixel_out,
  OutputColorspace output_colorspace,
  const int downsample_to_minimum_dimension,
  float * applied_scale_factor_out)
{
  jpeg_decompress_struct cinfo;
  my_error_mgr error_mgr;

  width_out = 0;
  height_out = 0;
  bytes_per_pixel_out = 0;

  // Set the normal JPEG error routines, and then override the error_exit function
  cinfo.err = jpeg_std_error(&error_mgr.jpeg);
  error_mgr.jpeg.error_exit = my_error_exit;

  // Set up the setjmp return context to use if the JPEG library encounters an error
  if (setjmp(error_mgr.setjmp_buffer))
  {
    // The JPEG library has encountered an error
    jpeg_destroy_decompress(&cinfo);
    return Failed;
  }

  // Initialize the JPEG decompression object
  jpeg_create_decompress(&cinfo);

  // This seems to have very little impact on the speed
  cinfo.dct_method = JDCT_FASTEST;

  // Set the input data source
  jpeg_mem_src(&cinfo, (unsigned char *)jpeg_data, jpeg_data_size);

  // Read file parameters
  jpeg_read_header(&cinfo, TRUE);

  // Set the output colorspace
  switch (output_colorspace)
  {
    case Original:
      break;
    case RGB:
      cinfo.out_color_space = JCS_RGB;
      break;
    case Grayscale:
      cinfo.out_color_space = JCS_GRAYSCALE;
      break;
  }

  // Check to see if a downsample_to_minimum_dimension value has been provided.
  if (downsample_to_minimum_dimension > 0)
  {
    // Compute the scale factors required to make the JPEG conform to the requested
    // image dimension.
    const float width_scale = float(cinfo.image_width) / downsample_to_minimum_dimension;
    const float height_scale = float(cinfo.image_height) / downsample_to_minimum_dimension;

    // Determine which scale is smaller, as that is the dimension that limits how much
    // the image can be resized while still maintaining the minimum requested image
    // dimension of downsample_to_minimum_dimension.
    const float min_scale = width_scale < height_scale ? width_scale : height_scale;

    if (min_scale < 2.0f)
    {
      // No-op, the image can't be downsized without making it smaller than the
      // requested image dimension of downsample_to_minimum_dimension.
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f;
      }
    }
    else if (min_scale < 4.0f)
    {
      // The image needs to be downsampled by a factor less than 4, so downsample
      // by the next-best factor, which is 2.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 2;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 2.0f;
      }
    }
    else if (min_scale < 8.0f)
    {
      // The image needs to be downsampled by a factor less than 8, so downsample
      // by the next-best factor, which is 4.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 4;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 4.0f;
      }
    }
    else
    {
      // The image needs to be downsampled by a factor of 8 or more, so downsample
      // by a factor of 8.
      cinfo.scale_num = 1;
      cinfo.scale_denom = 8;
      if (applied_scale_factor_out != NULL)
      {
        *applied_scale_factor_out = 1.0f / 8.0f;
      }
    }
  }
  else if (applied_scale_factor_out != NULL)
  {
    *applied_scale_factor_out = 1.0f;
  }

  // Start decompression
  jpeg_start_decompress(&cinfo);

  width_out = cinfo.output_width;
  height_out = cinfo.output_height;
  bytes_per_pixel_out = cinfo.output_components;

  // Check that there is sufficient preallocated data
  const int image_data_size = cinfo.output_width * cinfo.output_height * cinfo.output_components;
  if (image_data_size > preallocated_image_data_size)
  {
    // Don't print an error message to allow the caller to detect the failed
    // decode, resize their image buffer, and try again.
    //std::cerr << "jpeg_parser ERROR: not enough preallocated memory," << std::endl;
    //std::cerr << "  requested " << image_data_size << " bytes, but "
    //          << preallocated_image_data_size << " bytes available" << std::endl;
    jpeg_destroy_decompress(&cinfo);
    return ResizeRequired;
  }

  // Read the JPEG image, one row at a time
  const unsigned int row_stride = cinfo.output_width * cinfo.output_components;
  unsigned char * image_data_ptr = preallocated_image_data;
  while (cinfo.output_scanline < cinfo.output_height)
  {
    // jpeg_read_scanlines expects an array of pointers to scanlines, so it
    // is easiest to just provide a pointer to the current row and read
    // one row at a time
    jpeg_read_scanlines(&cinfo, (JSAMPARRAY)(&image_data_ptr), 1);
    image_data_ptr += row_stride;
  }

  // Finish decompression
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  return Success;
}

#include <image_resizer/image_resizer.h>
#include <core/colorspace_opencv.h>
#include <opencv2/opencv.hpp>
#include <iostream>

image_resizer::ImageResizer::ImageResizer(
  const image_resizer::ImageResizer::ResizeMode resize_mode,
  const int resized_image_dimension)
: m_resize_mode(resize_mode),
  m_resized_image_dimension(resized_image_dimension)
{}

image_resizer::ImageResizer::~ImageResizer()
{}

image_resizer::ImageResizer::ResizeResult image_resizer::ImageResizer::resize_image(
  const unsigned char * const input_image_data,
  const int input_image_width,
  const int input_image_height,
  const core::colorspace::Colorspace input_image_colorspace,
  unsigned char * const output_preallocated_image_data,
  const int output_preallocated_image_data_size)
{
  if (m_resize_mode == Resized)
  {
    // Test to see if the image needs to be resized.
    if (input_image_width > m_resized_image_dimension ||
      input_image_height > m_resized_image_dimension)
    {
      // The image needs to be resized, so create an OpenCV wrapper around
      // the input image.
      const cv::Mat input_image(
        input_image_height, // Number of rows.
        input_image_width, // Number of columns.
        core::colorspace::convert_to_opencv(input_image_colorspace),
        const_cast<unsigned char *>(input_image_data));

      ResizeResult result;
      // Compute the new (resized) width and height of the image.
      if (input_image_width > input_image_height)
      {
        result.scale = float(m_resized_image_dimension) / float(input_image_width);
        result.width = m_resized_image_dimension;
        result.height = static_cast<int>(result.scale * float(input_image_height) + 0.5f);
      }
      else // if (input_image_width <= input_image_height)
      {
        result.scale = float(m_resized_image_dimension) / float(input_image_height);
        result.height = m_resized_image_dimension;
        result.width = static_cast<int>(result.scale * float(input_image_width) + 0.5f);
      }

      const int required_data_size = result.width * result.height *
        core::colorspace::num_bytes_per_pixel(input_image_colorspace);
      if (required_data_size > output_preallocated_image_data_size)
      {
        std::cerr << "ERROR: ImageResizer::resize_image() - not enough preallocated image data" << std::endl;
        result.clear();
        return result;
      }

      // Create an OpenCV wrapper for the final image.
      cv::Mat resized_image(
        result.height,
        result.width,
        core::colorspace::convert_to_opencv(input_image_colorspace),
        output_preallocated_image_data);

      // Resize the image.
      cv::resize(
        input_image,
        resized_image,
        cv::Size(result.width, result.height),
        0, 0,
        CV_INTER_AREA);

      return result;
    }
    else // if (input_image_width <= m_resized_image_dimension &&
         //     input_image_height <= m_resized_image_dimension)
    {
      // The image does not need to be resized.

      ResizeResult result;
      result.width = input_image_width;
      result.height = input_image_height;
      result.scale = 1.0f;

      const int required_data_size = input_image_width * input_image_height *
        core::colorspace::num_bytes_per_pixel(input_image_colorspace);
      if (required_data_size > output_preallocated_image_data_size)
      {
        std::cerr << "ERROR: ImageResizer::resize_image() - not enough preallocated image data" << std::endl;
        result.clear();
        return result;
      }

      // The image does not need to be resized, so copy it into the image buffer.
      memcpy(
        output_preallocated_image_data, // destination
        input_image_data, // source
        required_data_size); // num bytes

      return result;
    }
  } // end (m_resize_mode == Resized)
  else if (m_resize_mode == ResizedExact)
  {
    if (input_image_width > m_resized_image_dimension ||
      input_image_height > m_resized_image_dimension)
    {
      // The image needs to be downsampled, so create an OpenCV wrapper around the loaded image.
      const cv::Mat input_image(
        input_image_height,
        input_image_width,
        core::colorspace::convert_to_opencv(input_image_colorspace),
        const_cast<unsigned char *>(input_image_data));

      ResizeResult result;
      // Compute the new (resized) width and height of the image.
      if (input_image_width > input_image_height)
      {
        result.scale = float(m_resized_image_dimension) / float(input_image_width);
        result.width = m_resized_image_dimension;
        result.height = static_cast<int>(result.scale * float(input_image_height) + 0.5f);
      }
      else // if (input_image_width <= input_image_height)
      {
        result.scale = float(m_resized_image_dimension) / float(input_image_height);
        result.height = m_resized_image_dimension;
        result.width = static_cast<int>(result.scale * float(input_image_width) + 0.5f);
      }

      const int required_data_size = result.width * result.height *
        core::colorspace::num_bytes_per_pixel(input_image_colorspace);
      if (required_data_size > output_preallocated_image_data_size)
      {
        std::cerr << "ERROR: ImageResizer::resize_image() - not enough preallocated image data" << std::endl;
        result.clear();
        return result;
      }

      // Create an OpenCV wrapper for the final image.
      cv::Mat resized_image(
        result.height,
        result.width,
        core::colorspace::convert_to_opencv(input_image_colorspace),
        output_preallocated_image_data);

      // Downsample the image.
      cv::resize(
        input_image,
        resized_image,
        cv::Size(result.width, result.height),
        0, 0,
        CV_INTER_AREA);

      return result;
    }
    else if (input_image_width < m_resized_image_dimension &&
      input_image_height < m_resized_image_dimension)
    {
      // The image needs to be upsampled, so create an OpenCV wrapper around the loaded image.
      cv::Mat input_image(
        input_image_height,
        input_image_width,
        core::colorspace::convert_to_opencv(input_image_colorspace),
        const_cast<unsigned char *>(input_image_data));

      ResizeResult result;
      // Compute the new (resized) width and height of the image.
      if (input_image_width > input_image_height)
      {
        result.scale = float(m_resized_image_dimension) / float(input_image_width);
        result.width = m_resized_image_dimension;
        result.height = static_cast<int>(result.scale * float(input_image_height) + 0.5f);
      }
      else // if (input_image_width <= input_image_height)
      {
        result.scale = float(m_resized_image_dimension) / float(input_image_height);
        result.height = m_resized_image_dimension;
        result.width = static_cast<int>(result.scale * float(input_image_width) + 0.5f);
      }

      const int required_data_size = result.width * result.height *
        core::colorspace::num_bytes_per_pixel(input_image_colorspace);
      if (required_data_size > output_preallocated_image_data_size)
      {
        std::cerr << "ERROR: ImageResizer::resize_image() - not enough preallocated image data" << std::endl;
        result.clear();
        return result;
      }

      // Create an OpenCV wrapper for the final image.
      cv::Mat resized_image(
        result.height,
        result.width,
        core::colorspace::convert_to_opencv(input_image_colorspace),
        output_preallocated_image_data);

      // Upsample the image.
      cv::resize(
        input_image,
        resized_image,
        cv::Size(result.width, result.height),
        0, 0,
        CV_INTER_LINEAR);

      return result;
    }
    else
    {
      // The image does not need to be resized.

      ResizeResult result;
      result.width = input_image_width;
      result.height = input_image_height;
      result.scale = 1.0f;

      const int required_data_size = input_image_width * input_image_height *
        core::colorspace::num_bytes_per_pixel(input_image_colorspace);
      if (required_data_size > output_preallocated_image_data_size)
      {
        std::cerr << "ERROR: ImageResizer::resize_image() - not enough preallocated image data" << std::endl;
        result.clear();
        return result;
      }

      // The image does not need to be resized, so copy it into the image buffer.
      memcpy(
        output_preallocated_image_data, // destination
        input_image_data, // source
        required_data_size); // num bytes

      return result;
    }
  } // end (m_resize_mode == ResizedExact)
  else if (m_resize_mode == ResizedSquare)
  {
    // Create an OpenCV wrapper around the loaded image.
    cv::Mat input_image(
      input_image_height,
      input_image_width,
      core::colorspace::convert_to_opencv(input_image_colorspace),
      const_cast<unsigned char *>(input_image_data));

    // Check if the image will need to be cropped.
    if (input_image_width != input_image_height)
    {
      // The image needs to be cropped, so compute the bounds of a centered, square image region.
      int row_start = 0;
      int row_end = input_image_height;
      int col_start = 0;
      int col_end = input_image_width;
      if (input_image_width > input_image_height)
      {
        const int diff = input_image_width - input_image_height;
        col_start = diff / 2;
        col_end = col_start + input_image_height;
      }
      else // (input_image_width <= input_image_height)
      {
        const int diff = input_image_height - input_image_width;
        row_start = diff / 2;
        row_end = row_start + input_image_width;
      }
      // Crop the image so that it is square.
      input_image = input_image(cv::Range(row_start, row_end), cv::Range(col_start, col_end));
    }

    // Get the dimension of the square image.
    const int image_dimension = input_image.rows;

    // Check if the image needs to be resized.
    if (image_dimension > m_resized_image_dimension)
    {
      ResizeResult result;
      result.width = m_resized_image_dimension;
      result.height = m_resized_image_dimension;
      result.scale = float(m_resized_image_dimension) / image_dimension;

      const int required_data_size = m_resized_image_dimension * m_resized_image_dimension *
        core::colorspace::num_bytes_per_pixel(input_image_colorspace);
      if (required_data_size > output_preallocated_image_data_size)
      {
        std::cerr << "ERROR: ImageResizer::resize_image() - not enough preallocated image data" << std::endl;
        result.clear();
        return result;
      }

      // The image needs to be resized, so create an OpenCV wrapper for the final image.
      cv::Mat resized_image(
        m_resized_image_dimension,
        m_resized_image_dimension,
        core::colorspace::convert_to_opencv(input_image_colorspace),
        output_preallocated_image_data);

      // Resize the image (which stores the result in the output buffer).
      cv::resize(
        input_image,
        resized_image,
        cv::Size(m_resized_image_dimension, m_resized_image_dimension),
        0, 0,
        CV_INTER_AREA);

      return result;
    }
    else // if (image_dimension <= m_resized_image_dimension)
    {
      ResizeResult result;
      result.width = image_dimension;
      result.height = image_dimension;
      result.scale = 1.0f;

      const int required_data_size = image_dimension * image_dimension *
        core::colorspace::num_bytes_per_pixel(input_image_colorspace);
      if (required_data_size > output_preallocated_image_data_size)
      {
        std::cerr << "ERROR: ImageResizer::resize_image() - not enough preallocated image data" << std::endl;
        result.clear();
        return result;
      }

      // The imgae does not need to be resized, so create an OpenCV wrapper for the final image.
      cv::Mat final_image(
        image_dimension,
        image_dimension,
        core::colorspace::convert_to_opencv(input_image_colorspace),
        output_preallocated_image_data);

      // Copy the image to the output buffer.
      input_image.copyTo(final_image);

      return result;
    }
  } // end (m_resize_mode == ResizedSquare)
  else
  {
    std::cerr << "ERROR: ImageResizer::resize_image() - unrecognized option" << std::endl;
    exit(EXIT_FAILURE);
  }
}

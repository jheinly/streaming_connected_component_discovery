#pragma once
#ifndef IMAGE_RESIZER_H
#define IMAGE_RESIZER_H

#include <core/colorspace.h>

namespace image_resizer {

class ImageResizer
{
  public:
    enum ResizeMode
    {
      Resized,      // All output images will be resized so that their longest
                    // dimension is at or below a certain threshold.
      ResizedExact, // All output images will be resized so that their longest
                    // dimension is exactly equal to a certain threshold.
      ResizedSquare // All output images will be cropped so that they are square
                    // and then resized so that their longest dimension is at or
                    // below a certain threshold.
    };

    struct ResizeResult
    {
      ResizeResult()
      : width(0),
        height(0),
        scale(0.0f)
      {}

      void clear()
      {
        width = 0;
        height = 0;
        scale = 0.0f;
      }

      int width;
      int height;
      float scale;
    };

    ImageResizer(
      const ResizeMode resize_mode,
      const int resized_image_dimension);

    ~ImageResizer();

    ResizeResult resize_image(
      const unsigned char * const input_image_data,
      const int input_image_width,
      const int input_image_height,
      const core::colorspace::Colorspace input_image_colorspace,
      unsigned char * const output_preallocated_image_data,
      const int output_preallocated_image_data_size);

    const int resized_image_dimension() const
    { return m_resized_image_dimension; }

  private:
    ImageResizer(const ImageResizer &);
    ImageResizer & operator=(const ImageResizer &);

    const ResizeMode m_resize_mode;
    const int m_resized_image_dimension;
};

} // namespace image_resizer

#endif // IMAGE_RESIZER_H

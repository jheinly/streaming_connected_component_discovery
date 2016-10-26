#pragma once
#ifndef STREAMING_IMAGE_LOADER_H
#define STREAMING_IMAGE_LOADER_H

#include <vector>
#include <buffer_data/file_data.h>
#include <buffer_data/image_data.h>
#include <core/colorspace.h>
#include <core/shared_batch_buffer.h>
#include <core/shared_speed_stats.h>
#include <core/streaming_module_interface.h>
#include <image_loader/image_loader.h>
#include <boost/thread.hpp>

class StreamingImageLoader : public core::StreamingModuleInterface
{
  public:
    explicit StreamingImageLoader(
      const int num_threads,
      core::SharedBatchBuffer<buffer_data::FileData> * input_file_data_buffer,
      core::SharedBatchBuffer<buffer_data::ImageData> * output_image_data_buffer,
      const core::colorspace::Colorspace output_image_colorspace,
      const image_resizer::ImageResizer::ResizeMode output_image_resize_mode,
      const int resized_image_dimension,
      const int minimum_allowed_image_dimension = 1,
      const float maximum_allowed_aspect_ratio = 0.0f,
      const int maximum_loaded_image_dimension_hint = 0);

    ~StreamingImageLoader();

    void generate_summary_speed_report(std::string & report);
    void generate_detailed_speed_report(std::string & report);

    void wait_until_finished();

  private:
    StreamingImageLoader(const StreamingImageLoader &);
    StreamingImageLoader & operator=(const StreamingImageLoader &);

    void thread_run(const int thread_num);

    // Input file buffer readers.
    const core::SharedBatchBuffer<buffer_data::FileData> * m_input_file_data_buffer;
    std::vector<boost::shared_ptr<core::SharedBatchBufferLockstepReader<buffer_data::FileData> > > m_file_data_readers;

    // Output image buffer writers.
    const core::SharedBatchBuffer<buffer_data::ImageData> * m_output_image_data_buffer;
    std::vector<boost::shared_ptr<core::SharedBatchBufferLockstepWriter<buffer_data::ImageData> > > m_image_data_writers;

    // Thread information.
    boost::thread * m_threads; // A std::vector of threads doesn't compile on Mac.
    const int m_num_threads;
    std::vector<image_loader::ImageLoader *> m_image_loaders;

    // Image settings.
    const core::colorspace::Colorspace m_output_image_colorspace;
    const image_resizer::ImageResizer::ResizeMode m_output_image_resize_mode;
    const int m_resized_image_dimension;
    const int m_minimum_allowed_image_dimension;
    const float m_maximum_allowed_aspect_ratio;
    const int m_maximum_loaded_image_dimension_hint;
    int m_maximum_image_size;
};

#endif // STREAMING_IMAGE_LOADER_H

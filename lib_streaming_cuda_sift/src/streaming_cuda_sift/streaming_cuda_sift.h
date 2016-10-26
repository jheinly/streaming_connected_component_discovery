#pragma once
#ifndef STREAMING_CUDA_SIFT_H
#define STREAMING_CUDA_SIFT_H

#include <buffer_data/image_data.h>
#include <buffer_data/sift_features_data.h>
#include <core/shared_batch_buffer.h>
#include <core/shared_speed_stats.h>
#include <core/streaming_module_interface.h>
#include <cuda_sift/cuda_sift.h>
#include <boost/thread.hpp>
#include <vector>
#include <string>

class StreamingCudaSift : public core::StreamingModuleInterface
{
  public:
    explicit StreamingCudaSift(
      const int num_threads,
      const std::vector<int> & gpu_nums,
      core::SharedBatchBuffer<buffer_data::ImageData> * input_image_data_buffer,
      core::SharedBatchBuffer<buffer_data::SiftFeaturesData> * output_sift_features_data_buffer,
      const int max_image_dimension,
      const int max_num_features,
      const int min_allowed_num_features = 1);

    ~StreamingCudaSift();

    void generate_summary_speed_report(std::string & report);
    void generate_detailed_speed_report(std::string & report);

    void wait_until_finished();

  private:
    StreamingCudaSift(const StreamingCudaSift &);
    StreamingCudaSift & operator=(const StreamingCudaSift &);

    void thread_run(const int thread_num);

    // Input image readers.
    const core::SharedBatchBuffer<buffer_data::ImageData> * m_input_image_data_buffer;
    std::vector<boost::shared_ptr<core::SharedBatchBufferLockstepReader<buffer_data::ImageData> > > m_image_data_readers;

    // Output sift features writers.
    const core::SharedBatchBuffer<buffer_data::SiftFeaturesData> * m_output_sift_features_data_buffer;
    std::vector<boost::shared_ptr<core::SharedBatchBufferLockstepWriter<buffer_data::SiftFeaturesData> > > m_sift_features_data_writers;

    // Thread information.
    boost::thread * m_threads; // A std::vector of threads doesn't compile on Mac.
    const int m_num_threads;
    std::vector<int> m_gpu_nums;
    std::vector<std::string> m_gpu_names;
    core::SharedSpeedStats * m_cuda_sifts_speed_stats;

    // Sift settings.
    const int m_max_image_dimension;
    const int m_max_num_features;
    const int m_min_allowed_num_features;
};

#endif // STREAMING_CUDA_SIFT_H

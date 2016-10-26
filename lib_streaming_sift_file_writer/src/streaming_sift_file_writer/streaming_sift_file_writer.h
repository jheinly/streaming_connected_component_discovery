#pragma once
#ifndef STREAMING_SIFT_FILE_WRITER_H
#define STREAMING_SIFT_FILE_WRITER_H

#include <buffer_data/output_sift_features_data.h>
#include <core/shared_batch_buffer.h>
#include <core/streaming_module_interface.h>
#include <v3d_support/pcdb.h>
#include <boost/thread.hpp>
#include <vector>

class StreamingSiftFileWriter : public core::StreamingModuleInterface
{
  public:
    explicit StreamingSiftFileWriter(
      const int num_threads,
      const v3d_support::PCDB * pcdb,
      core::SharedBatchBuffer<buffer_data::OutputSiftFeaturesData> * output_sift_data_buffer);

    ~StreamingSiftFileWriter();

    void generate_summary_speed_report(std::string & report);
    void generate_detailed_speed_report(std::string & report);

    void wait_until_finished();

  private:
    StreamingSiftFileWriter(const StreamingSiftFileWriter &);
    StreamingSiftFileWriter & operator=(const StreamingSiftFileWriter &);

    void thread_run(const int thread_num);

    // Input image readers.
    const core::SharedBatchBuffer<buffer_data::OutputSiftFeaturesData> * m_output_sift_data_buffer;
    std::vector<boost::shared_ptr<core::SharedBatchBufferLockstepReader<buffer_data::OutputSiftFeaturesData> > > m_output_sift_data_readers;

    // Thread information.
    boost::thread * m_threads; // A std::vector of threads doesn't compile on Mac.
    const int m_num_threads;

    const v3d_support::PCDB * m_pcdb;
};

#endif // STREAMING_SIFT_FILE_WRITER_H

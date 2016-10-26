#pragma once
#ifndef STREAMING_VISUAL_WORDS_H
#define STREAMING_VISUAL_WORDS_H

#include <buffer_data/sift_features_data.h>
#include <buffer_data/sift_visual_words_data.h>
#include <core/shared_batch_buffer.h>
#include <core/streaming_module_interface.h>
#include <vocab_tree/VocabTree.h>
#include <boost/thread.hpp>
#include <vector>

class StreamingVisualWords : public core::StreamingModuleInterface
{
  public:
    explicit StreamingVisualWords(
      const int num_threads,
      const VocabTree * vocab_tree,
      core::SharedBatchBuffer<buffer_data::SiftFeaturesData> * input_sift_features_data_buffer,
      core::SharedBatchBuffer<buffer_data::SiftVisualWordsData> * output_visual_words_data_buffer);

    ~StreamingVisualWords();

    void generate_summary_speed_report(std::string & report);
    void generate_detailed_speed_report(std::string & report);

    void wait_until_finished();

  private:
    StreamingVisualWords(const StreamingVisualWords &);
    StreamingVisualWords & operator=(const StreamingVisualWords &);

    void thread_run(const int thread_num);

    // Input image readers.
    const core::SharedBatchBuffer<buffer_data::SiftFeaturesData> * m_input_sift_features_data_buffer;
    std::vector<boost::shared_ptr<core::SharedBatchBufferLockstepReader<buffer_data::SiftFeaturesData> > > m_sift_features_data_readers;

    // Output sift features writers.
    const core::SharedBatchBuffer<buffer_data::SiftVisualWordsData> * m_output_visual_words_data_buffer;
    std::vector<boost::shared_ptr<core::SharedBatchBufferLockstepWriter<buffer_data::SiftVisualWordsData> > > m_visual_words_data_writers;

    // Thread information.
    boost::thread * m_threads; // A std::vector of threads doesn't compile on Mac.
    const int m_num_threads;

    const VocabTree * m_vocab_tree;
};

#endif // STREAMING_VISUAL_WORDS_H

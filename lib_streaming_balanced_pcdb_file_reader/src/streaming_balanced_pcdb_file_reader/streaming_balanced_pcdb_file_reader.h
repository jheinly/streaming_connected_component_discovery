#pragma once
#ifndef STREAMING_BALANCED_PCDB_FILE_READER_H
#define STREAMING_BALANCED_PCDB_FILE_READER_H

#include <buffer_data/file_data.h>
#include <core/manager_worker_synchronization.h>
#include <core/shared_batch_buffer.h>
#include <core/shared_speed_stats.h>
#include <core/shared_value.h>
#include <core/streaming_module_interface.h>
#include <v3d_support/pcdb.h>
#include <boost/thread.hpp>
#include <string>
#include <vector>

class StreamingBalancedPcdbFileReader : public core::StreamingModuleInterface
{
  public:
    explicit StreamingBalancedPcdbFileReader(
      const v3d_support::PCDB * pcdb,
      const v3d_support::PCDB::FileType file_type,
      const std::vector<std::string> & file_list_names, // the number of lists determines the number of threads
      core::SharedBatchBuffer<buffer_data::FileData> * output_file_data_buffer,
      const int num_files_to_skip = 0);

    ~StreamingBalancedPcdbFileReader();

    void generate_summary_speed_report(std::string & report);
    void generate_detailed_speed_report(std::string & report);

    void wait_until_finished();

    const core::SharedValue<int> * num_files_remaining() const
    { return &m_num_files_remaining; }

    const core::SharedValue<int> * total_num_files() const
    { return &m_total_num_files; }

  private:
    StreamingBalancedPcdbFileReader(const StreamingBalancedPcdbFileReader &);
    StreamingBalancedPcdbFileReader & operator=(const StreamingBalancedPcdbFileReader &);

    void master_thread_run();
    void worker_thread_run(const int thread_num);

    // Output file buffer writer.
    const core::SharedBatchBuffer<buffer_data::FileData> * m_output_file_data_buffer;
    boost::shared_ptr<core::SharedBatchBufferLockstepWriter<buffer_data::FileData> > m_file_data_writer;

    boost::thread m_master_thread;
    boost::thread * m_worker_threads;
    const int m_num_worker_threads;

    const v3d_support::PCDB * m_pcdb;
    const v3d_support::PCDB::FileType m_file_type;

    core::ManagerWorkerSynchronization m_synchronization;

    std::vector<std::string> m_file_list_names;
    core::SharedSpeedStats * m_file_lists_speed_stats;
    int m_total_num_file_list_entries;
    std::vector<int> m_num_entries_remaining_per_file_list;
    std::vector<int> m_num_entries_to_read_per_file_list;
    std::vector<int> m_current_file_index_per_file_list;
    std::vector<int> m_num_entries_to_skip_per_file_list;

    core::SharedValue<int> m_num_files_remaining;
    core::SharedValue<int> m_total_num_files;
    core::SharedValue<bool> m_done_with_all_reads;
    const int m_num_files_to_skip;
};

#endif // STREAMING_BALANCED_PCDB_FILE_READER_H

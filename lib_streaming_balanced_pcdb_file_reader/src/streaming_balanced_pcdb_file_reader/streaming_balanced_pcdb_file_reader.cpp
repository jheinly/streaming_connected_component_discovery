#include <streaming_balanced_pcdb_file_reader/streaming_balanced_pcdb_file_reader.h>
#include <core/file_helper.h>
#include <core/timer.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>

StreamingBalancedPcdbFileReader::StreamingBalancedPcdbFileReader(
  const v3d_support::PCDB * pcdb,
  const v3d_support::PCDB::FileType file_type,
  const std::vector<std::string> & file_list_names,
  core::SharedBatchBuffer<buffer_data::FileData> * output_file_data_buffer,
  const int num_files_to_skip)
: core::StreamingModuleInterface(1), // 1 thread (the master thread) will interact with the shared buffer
  m_output_file_data_buffer(output_file_data_buffer),
  m_file_data_writer(),
  m_master_thread(),
  m_worker_threads(NULL),
  m_num_worker_threads(static_cast<int>(file_list_names.size())),
  m_pcdb(pcdb),
  m_file_type(file_type),
  m_synchronization(static_cast<int>(file_list_names.size())), // number of workers
  m_file_list_names(file_list_names),
  m_file_lists_speed_stats(NULL),
  m_total_num_file_list_entries(0),
  m_num_entries_remaining_per_file_list(static_cast<int>(file_list_names.size()), 0),
  m_num_entries_to_read_per_file_list(static_cast<int>(file_list_names.size()), 0),
  m_current_file_index_per_file_list(static_cast<int>(file_list_names.size()), 0),
  m_num_entries_to_skip_per_file_list(static_cast<int>(file_list_names.size()), 0),
  m_num_files_remaining(0),
  m_total_num_files(0),
  m_done_with_all_reads(false),
  m_num_files_to_skip(num_files_to_skip)
{
  m_file_data_writer = output_file_data_buffer->get_new_lockstep_writer();

  m_file_lists_speed_stats = new core::SharedSpeedStats[m_num_worker_threads];

  m_master_thread = boost::thread(&StreamingBalancedPcdbFileReader::master_thread_run, this);

  m_worker_threads = new boost::thread[m_num_worker_threads];
  for (int i = 0; i < m_num_worker_threads; ++i)
  {
    m_worker_threads[i] = boost::thread(&StreamingBalancedPcdbFileReader::worker_thread_run, this, i);
  }
}

StreamingBalancedPcdbFileReader::~StreamingBalancedPcdbFileReader()
{
  if (m_file_lists_speed_stats != NULL)
  {
    delete [] m_file_lists_speed_stats;
    m_file_lists_speed_stats = NULL;
  }
  if (m_worker_threads != NULL)
  {
    delete [] m_worker_threads;
    m_worker_threads = NULL;
  }
}

void StreamingBalancedPcdbFileReader::generate_summary_speed_report(std::string & report)
{
  const core::SharedSpeedStats * file_writers_speed_stats =
    m_output_file_data_buffer->lockstep_writers_speed_stats();

  std::stringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream.precision(1);
  stream << "StreamingBalancedPcdbFileReader" << std::endl;
  stream << "  Output:           ";
  stream.width(7);
  stream << file_writers_speed_stats->get_most_recent_speed_hz() << " Hz";
  stream << "      (rolling avg: ";
  stream.width(7);
  stream << file_writers_speed_stats->get_rolling_average_speed_hz() << " Hz)";
  stream << "      (overall avg: ";
  stream.width(7);
  stream << file_writers_speed_stats->get_overall_average_speed_hz() << " Hz)";
  stream << std::endl;

  report = stream.str();
}

void StreamingBalancedPcdbFileReader::generate_detailed_speed_report(std::string & report)
{
  std::stringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream.precision(1);

  generate_summary_speed_report(report);
  stream << report;

  for (int i = 0; i < m_num_worker_threads; ++i)
  {
    stream << "    - " << m_file_list_names[i] << std::endl;
    stream << "                    ";
    stream.width(7);
    stream << m_file_lists_speed_stats[i].get_most_recent_speed_hz() << " Hz";
    stream << "      (rolling avg: ";
    stream.width(7);
    stream << m_file_lists_speed_stats[i].get_rolling_average_speed_hz() << " Hz)";
    stream << "      (overall avg: ";
    stream.width(7);
    stream << m_file_lists_speed_stats[i].get_overall_average_speed_hz() << " Hz)";
    stream << std::endl;
  }

  report = stream.str();
}

void StreamingBalancedPcdbFileReader::wait_until_finished()
{
  for (int i = 0; i < m_num_worker_threads; ++i)
  {
    m_worker_threads[i].join();
  }
  m_master_thread.join();
}

void StreamingBalancedPcdbFileReader::master_thread_run()
{
  // Wait for the worker threads to scan their file lists.
  m_synchronization.wait_for_workers_to_be_ready_to_be_signaled();

  int total_num_file_list_entries_remaining = 0;
  for (int i = 0; i < m_num_worker_threads; ++i)
  {
    m_current_file_index_per_file_list[i] = total_num_file_list_entries_remaining;
    total_num_file_list_entries_remaining += m_num_entries_remaining_per_file_list[i];
  }

  m_num_files_remaining.set(total_num_file_list_entries_remaining);
  m_total_num_files.set(total_num_file_list_entries_remaining);

  std::cout << std::endl;
  std::cout << "StreamingBalancedPcdbFileReader" << std::endl;
  for (int i = 0; i < m_num_worker_threads; ++i)
  {
    std::cout << "  " << m_file_list_names[i] << std::endl;
    std::cout << "    " << m_num_entries_remaining_per_file_list[i] << " files" << std::endl;
  }
  std::cout << std::endl;

  if (m_num_files_to_skip > 0)
  {
    std::cout << "StreamingBalancedPcdbFileReader: attempting to skip " <<
      m_num_files_to_skip << " files..." << std::endl;

    ASSERT(m_num_files_to_skip < total_num_file_list_entries_remaining);

    int num_skipped_files = 0;
    for (int i = 0; i < m_num_worker_threads; ++i)
    {
      const int num_entries_to_skip = static_cast<int>(
        static_cast<double>(m_num_files_to_skip) *
        static_cast<double>(m_num_entries_remaining_per_file_list[i]) /
        static_cast<double>(total_num_file_list_entries_remaining));
      m_num_entries_to_skip_per_file_list[i] = num_entries_to_skip;
      m_num_entries_remaining_per_file_list[i] -= num_entries_to_skip;
      m_current_file_index_per_file_list[i] += num_entries_to_skip;
      num_skipped_files += num_entries_to_skip;
    }

    total_num_file_list_entries_remaining -= num_skipped_files;
    m_num_files_remaining.set(total_num_file_list_entries_remaining);

    std::cout << "StreamingBalancedPcdbFileReader: skipped " <<
      num_skipped_files << " files" << std::endl;
  }

  // Wait until this streaming module should start.
  worker_thread_wait_for_start();

  std::vector<std::pair<double, int> > num_entries_to_read_per_file_list_remainder(m_num_worker_threads);

  // Loop through all of the files in the file lists.
  for (;;)
  {
    m_file_data_writer->wait_for_next_write_buffer();

    // Compute the total number of remaining files to read.
    total_num_file_list_entries_remaining = 0;
    for (int i = 0; i < m_num_worker_threads; ++i)
    {
      total_num_file_list_entries_remaining += m_num_entries_remaining_per_file_list[i];
    }

    m_num_files_remaining.set(total_num_file_list_entries_remaining);

    // If there are no further files to read, quit.
    if (total_num_file_list_entries_remaining == 0)
    {
      m_done_with_all_reads.set(true);
      m_synchronization.signal_workers();
      m_file_data_writer->done_with_all_further_writes_and_wait_for_exit();
      return;
    }

    m_file_data_writer->starting_to_write_to_buffer();

    // Compute the number of files to read per file list.

    // First compute the integer and frational values of files that each thread
    // should read, and initially assign each thread the integer value of files.
    // This will either equal (or most likely be less than) the batch size.
    const int batch_size = static_cast<int>(m_file_data_writer->write_buffer().size());
    int num_unassigned_batch_entries = batch_size;
    for (int i = 0; i < m_num_worker_threads; ++i)
    {
      const double num_entries_to_read = static_cast<double>(batch_size) *
        static_cast<double>(m_num_entries_remaining_per_file_list[i]) /
        static_cast<double>(total_num_file_list_entries_remaining);
      m_num_entries_to_read_per_file_list[i] = static_cast<int>(num_entries_to_read);
      num_entries_to_read_per_file_list_remainder[i] = std::make_pair(
        num_entries_to_read - static_cast<double>(m_num_entries_to_read_per_file_list[i]),
        i);
      num_unassigned_batch_entries -= m_num_entries_to_read_per_file_list[i];
    }

    // Then, if the entire batch is not full from using the integer values,
    // sort the frational values from largest to smallest and assign the remaining
    // batch files to the threads in this order.
    if (num_unassigned_batch_entries > 0)
    {
      std::sort(
        num_entries_to_read_per_file_list_remainder.begin(),
        num_entries_to_read_per_file_list_remainder.end(),
        std::greater<std::pair<double, int> >());
      for (int i = 0; i < m_num_worker_threads; ++i)
      {
        if (num_entries_to_read_per_file_list_remainder[i].first <= 0)
        {
          break;
        }
        ++m_num_entries_to_read_per_file_list[num_entries_to_read_per_file_list_remainder[i].second];
        --num_unassigned_batch_entries;
        if (num_unassigned_batch_entries == 0)
        {
          break;
        }
      }
    }

    // Signal the workers to start reading from their file lists.
    m_synchronization.signal_workers();

    // Wait for the workers to finish reading their file lists.
    m_synchronization.wait_for_workers_to_be_ready_to_be_signaled();

    m_file_data_writer->done_writing_to_buffer();
  }
}

void StreamingBalancedPcdbFileReader::worker_thread_run(const int thread_num)
{
  const std::string & file_list_name = m_file_list_names[thread_num];
  FILE * file_list = core::file_helper::open_file(file_list_name, "r");

  const int line_size = 256;
  char line[line_size];

  const int format_size = 32;
  char format[format_size];
#ifdef WIN32
  sprintf_s(format, format_size, "%%%ds\n", line_size - 1);
#else
  sprintf(format, "%%%ds\n", line_size - 1);
#endif

  // Count the number of lines in the file.
  int num_lines = 0;
  for (;;)
  {
#ifdef WIN32
    const int num_read = fscanf_s(file_list, format, line, line_size);
#else
    const int num_read = fscanf(file_list, format, line);
#endif
    if (num_read == 1)
    {
      ++num_lines;
    }
    else
    {
      break;
    }
  }
  if (!feof(file_list))
  {
    std::cerr << "ERROR: format error in image list after reading " << num_lines << " lines," << std::endl;
    std::cerr << file_list_name << std::endl;
    fclose(file_list);
    exit(EXIT_FAILURE);
  }
  m_num_entries_remaining_per_file_list[thread_num] = num_lines;

  // Reset the file to the beginning.
  rewind(file_list);

  core::Timer timer;
  timer.stop();
  std::string file_path;
  for (;;)
  {
    m_synchronization.wait_for_signal_from_manager();

    if (m_done_with_all_reads.get())
    {
      fclose(file_list);
      return;
    }

    while (m_num_entries_to_skip_per_file_list[thread_num] > 0)
    {
      // Get the next line from the file list.
#ifdef WIN32
      const int num_read = fscanf_s(file_list, format, line, line_size);
#else
      const int num_read = fscanf(file_list, format, line);
#endif
      ALWAYS_ASSERT(num_read == 1);
      --m_num_entries_to_skip_per_file_list[thread_num];
    }

    timer.start();
    int num_entries_processed = 0;

    while (m_num_entries_to_read_per_file_list[thread_num] > 0 &&
           m_num_entries_remaining_per_file_list[thread_num] > 0)
    {
      // Get the next line from the file list.
#ifdef WIN32
      const int num_read = fscanf_s(file_list, format, line, line_size);
#else
      const int num_read = fscanf(file_list, format, line);
#endif
      ALWAYS_ASSERT(num_read == 1);
      --m_num_entries_remaining_per_file_list[thread_num];
      const int file_index = m_current_file_index_per_file_list[thread_num];
      ++m_current_file_index_per_file_list[thread_num];
      ++num_entries_processed;

      // Construct the path to the file.
      m_pcdb->get_indexed_file(m_file_type, line, file_path);

      // Make sure the path exists and corresponds to a non-zero sized file.
      const int file_size = core::file_helper::compute_file_size(file_path);
      if (file_size <= 0)
      {
        std::cerr << "WARNING: empty or non-existant file, " << file_path << std::endl;
        continue;
      }
      if (file_size > 50 * 1024 * 1024) // 50 MB
      {
        std::cerr << "WARNING: file is too large, " << file_path << std::endl;
        continue;
      }

      const int file_buffer_index =
        m_file_data_writer->write_buffer_counter()->get_value_and_increment();
      ASSERT(file_buffer_index < static_cast<int>(m_file_data_writer->write_buffer().size()));

      buffer_data::FileData & file_data = m_file_data_writer->write_buffer()[file_buffer_index];

      file_data.file_index = file_index;
      file_data.file_name = line;
      file_data.data.resize(file_size);

      FILE * file = core::file_helper::open_file(file_path, "rb");
      ALWAYS_ASSERT(file != NULL);

      size_t num_bytes_read = fread(&file_data.data[0], 1, file_size, file);
      ALWAYS_ASSERT(static_cast<int>(num_bytes_read) == file_size);
      fclose(file);

      --m_num_entries_to_read_per_file_list[thread_num];
    }

    timer.stop();

    m_file_lists_speed_stats[thread_num].add_timing(timer.elapsed(), num_entries_processed);
  }
}

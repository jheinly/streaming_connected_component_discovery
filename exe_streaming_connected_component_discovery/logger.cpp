#include "logger.h"

void init_logger(const std::string & log_folder)
{
  const std::string log_path = log_folder + "/console_log_%N.txt";
  const int rotation_size = 10 * 1024 * 1024; // 10 MB
  boost::log::add_file_log(
    boost::log::keywords::file_name = log_path,
    boost::log::keywords::rotation_size = rotation_size,
    boost::log::keywords::format = "%Message%");
  boost::log::add_console_log(
    std::cout,
    boost::log::keywords::format = "%Message%");
}

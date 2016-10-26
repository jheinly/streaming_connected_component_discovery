#include <core/file_helper.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <iostream>
#include <sys/stat.h>

FILE * core::file_helper::open_file(
  const std::string & file_path,
  const std::string & file_mode)
{
  FILE * file = NULL;
#ifdef WIN32
  fopen_s(&file, file_path.c_str(), file_mode.c_str());
#else
  file = fopen(file_path.c_str(), file_mode.c_str());
#endif
  
  if (file == NULL)
  {
    std::cerr << "ERROR: failed to open file," << std::endl;
    std::cerr << file_path << std::endl;
    throw(FileException(file_path));
  }
  
  return file;
}

void core::file_helper::open_file(
  const std::string & file_path,
  std::ifstream & input_file)
{
  input_file.open(file_path.c_str());
  if (input_file.fail())
  {
    std::cerr << "ERROR: failed to open file," << std::endl;
    std::cerr << file_path << std::endl;
    throw(FileException(file_path));
  }
}

void core::file_helper::open_file(
  const std::string & file_path,
  std::ofstream & output_file)
{
  output_file.open(file_path.c_str());
  if (output_file.fail())
  {
    std::cerr << "ERROR: failed to open file," << std::endl;
    std::cerr << file_path << std::endl;
    throw(FileException(file_path));
  }
}

std::string core::file_helper::trim_file_path(const std::string & file_path)
{
  std::string result = file_path;
  size_t slash = result.find_last_of("/\\");
  if (slash != std::string::npos)
  {
    result = result.substr(slash + 1);
  }

  size_t dot = result.find_first_of(".");
  if (dot != std::string::npos)
  {
    result = result.substr(0, dot);
  }

  return result;
}

int core::file_helper::compute_file_size(const std::string & file_path)
{
  struct stat status;
  const int error = stat(file_path.c_str(), &status);
  if (error == -1)
  {
    return 0;
  }
  return status.st_size;
}

bool core::file_helper::path_exists(const std::string & path)
{
  return boost::filesystem::exists(path);
}

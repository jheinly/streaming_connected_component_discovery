#pragma once
#ifndef FILE_DATA_H
#define FILE_DATA_H

#include <string>
#include <vector>

namespace buffer_data {

struct FileData
{
  FileData()
  : file_index(-1),
    file_name(),
    data()
  {}

  int file_index;
  std::string file_name;
  std::vector<unsigned char> data;
};

} // namespace buffer_data

#endif // FILE_DATA_H

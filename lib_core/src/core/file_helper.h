#pragma once
#ifndef FILE_HELPER_H
#define FILE_HELPER_H

#include <cstdio>
#include <string>
#include <fstream>
#include <exception>

namespace core {

namespace file_helper {

FILE * open_file(
  const std::string & file_path,
  const std::string & file_mode = "r");

void open_file(
  const std::string & file_path,
  std::ifstream & input_file);

void open_file(
  const std::string & file_path,
  std::ofstream & output_file);

std::string trim_file_path(const std::string & file_path);

int compute_file_size(const std::string & file_path);

bool path_exists(const std::string & path);

class FileException : public std::exception
{
  public:
    FileException(const std::string & message)
    : std::exception(),
      m_message(message)
    {}

    virtual const char * what() const throw()
    {
      return m_message.c_str();
    }

  private:
    std::string m_message;
};

} // namespace file_helper

} // namespace core

#endif // FILE_HELPER_H

// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#include "util/misc.h"

#include <fstream>
#include <sstream>

#include <boost/algorithm/string.hpp>

namespace colmap {

std::string ensure_trailing_slash(std::string str) {
  if (str.length() > 0) {
    if (str.at(str.length() - 1) != '/') {
      str += "/";
    }
  }
  return str;
}

void print_heading1(const std::string& heading) {
  std::cout << std::endl << std::string(78, '=') << std::endl;
  std::cout << heading << std::endl;
  std::cout << std::string(78, '=') << std::endl << std::endl;
}

void print_heading2(const std::string& heading) {
  std::cout << std::endl << heading << std::endl;
  std::cout << std::string(std::min<int>(heading.size(), 78), '-') << std::endl;
}

bool str_replace(std::string& str, const std::string& old_str,
                 const std::string& new_str) {
  const size_t start_pos = str.find(old_str);

  if (start_pos == std::string::npos) {
    return false;
  }

  str.replace(start_pos, old_str.length(), new_str);

  // Recursive replacement for multiple occurrences (inefficient)
  colmap::str_replace(str, old_str, new_str);

  return true;
}

std::vector<std::string> str_split(const std::string& str,
                                   const std::string& delim) {
  std::vector<std::string> elems;
  boost::split(elems, str, boost::is_any_of(delim), boost::token_compress_on);
  return elems;
}

}  // end namespace colmap

// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#ifndef COLMAP_SRC_UTIL_MISC_H_
#define COLMAP_SRC_UTIL_MISC_H_

#include <string>
#include <vector>
#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

namespace colmap {

/*
  Append trailing slash to string if it does not yet end in one.

  @param str    Input string.

  @return       Input string, with a trailing slash.
 */
std::string ensure_trailing_slash(std::string str);

/*
  Print first-order heading with over- and underscores to `std::cout`.

  @param heading      Heading text as a single line.
 */
void print_heading1(const std::string& heading);

/*
  Print second-order heading with underscores to `std::cout`.

  @param heading      Heading text as a single line.
 */
void print_heading2(const std::string& heading);

/*
  Replace all occurrences of `old_str` with `new_str` in the given input string.

  @param str      String to which to apply the replacement.
  @param old_str  Old string token for replacement.
  @param new_str  New string token for replacement.
 */
bool str_replace(std::string& str, const std::string& old_str,
                 const std::string& new_str);

/*
  Split string into list of words using the given delimiters.

  @param str    String to split.
  @param delim  Delimiters used to split the string. May contain multiple
                delimiters in the same string.

  @return       The words of the string.
 */
std::vector<std::string> str_split(const std::string& str,
                                   const std::string& delim);

template <typename T>
bool vector_contains(const std::vector<T>& vector, const T value) {
  return std::find_if(vector.begin(), vector.end(), [value](const T element) {
           return element == value;
         }) != vector.end();
}

/*
  Parse CSV line to a list of values.

  @param    The CSV string as a single line.

  @return   The elements of the CSV line.
 */
template <typename T>
std::vector<T> csv2vector(const std::string& csv) {
  auto elems = str_split(csv, ",;");
  std::vector<T> values;
  values.reserve(elems.size());
  for (auto& elem : elems) {
    boost::erase_all(elem, " ");
    if (elem.empty()) {
      continue;
    }
    try {
      values.push_back(boost::lexical_cast<T>(elem));
    } catch (std::exception) {
      return std::vector<T>(0);
    }
  }
  return values;
}

}  // end namespace colmap

#endif  // COLMAP_SRC_UTIL_MISC_H_

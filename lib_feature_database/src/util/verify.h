// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#ifndef COLMAP_SRC_UTIL_VERIFY_H_
#define COLMAP_SRC_UTIL_VERIFY_H_

#include <iostream>

#include <boost/assert.hpp>

namespace colmap {

#ifndef VERIFY
#define VERIFY(x) verify_value((x), __FILE__, __FUNCTION__, __LINE__)
#endif
#ifndef VERIFY_EQUAL
#define VERIFY_EQUAL(x, y) \
  verify_value((x) == (y), __FILE__, __FUNCTION__, __LINE__)
#endif
#ifndef VERIFY_ALMOST_EQUAL
#define VERIFY_ALMOST_EQUAL(x, y, eps) \
  verify_value(std::abs((x) - (y)) < (eps), __FILE__, __FUNCTION__, __LINE__)
#endif
#ifndef VERIFY_NOTNULL
#define VERIFY_NOTNULL(x) \
  verify_value((x != nullptr), __FILE__, __FUNCTION__, __LINE__)
#endif

#ifndef VERIFY_MSG
#define VERIFY_MSG(x, msg) \
  verify_value_msg((x), msg, __FILE__, __FUNCTION__, __LINE__)
#endif
#ifndef VERIFY_EQUAL_MSG
#define VERIFY_EQUAL_MSG(x, y, msg) \
  verify_value_msg((x) == (y), msg, __FILE__, __FUNCTION__, __LINE__)
#endif
#ifndef VERIFY_ALMOST_EQUAL_MSG
#define VERIFY_ALMOST_EQUAL_MSG(x, y, eps, msg)                              \
  verify_value_msg(std::abs((x) - (y)) < (eps), msg, __FILE__, __FUNCTION__, \
                   __LINE__)
#endif

void verify_value(const bool value, const std::string& file_name,
                  const std::string& func_name, const int line_number);

void verify_value_msg(const bool value, const std::string& msg,
                      const std::string& file_name,
                      const std::string& func_name, const int line_number);

}  // end namespace colmap

#endif  // COLMAP_SRC_UTIL_VERIFY_H_

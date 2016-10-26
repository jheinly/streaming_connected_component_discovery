// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#include "util/verify.h"

namespace colmap {

void verify_value(const bool value, const std::string& file_name,
                  const std::string& func_name, const int line_number) {
  if (!value) {
    printf("Verification failed [%s, %s, line %i]\n", file_name.c_str(),
           func_name.c_str(), line_number);
    exit(EXIT_FAILURE);
  }
}

void verify_value_msg(const bool value, const std::string& msg,
                      const std::string& file_name,
                      const std::string& func_name, const int line_number) {
  if (!value) {
    printf("Verification failed [%s, %s, line %i]: %s\n", file_name.c_str(),
           func_name.c_str(), line_number, msg.c_str());
    exit(EXIT_FAILURE);
  }
}

}  // end namespace colmap

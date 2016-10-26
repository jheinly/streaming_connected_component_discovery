// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#ifndef COLMAP_SRC_UTIL_SQLITE3_UTILS_
#define COLMAP_SRC_UTIL_SQLITE3_UTILS_

#include <string>
#include <cstdio>
#include <cstdlib>

#if 0 // Disabled as this code is not necessary for the feature database.
#include "ext/SQLite/sqlite3.h"
#else
#include <sqlite/sqlite3.h>
#endif // Disabled as this code is not necessary for the feature database.

namespace colmap {

inline int sqlite3_helper_call(const int result_code,
                               const std::string& filename,
                               const int line_number) {
  switch (result_code) {
    case SQLITE_OK:
    case SQLITE_ROW:
    case SQLITE_DONE:
      return result_code;
    default:
      printf("SQLite error [%s, line %i]: %s\n", filename.c_str(), line_number,
             sqlite3_errstr(result_code));
      exit(EXIT_FAILURE);
  }
}

#define SQLITE3_CALL(func) sqlite3_helper_call(func, __FILE__, __LINE__)

#define SQLITE3_EXEC(database, sql, callback)                                  \
  {                                                                            \
    char* err_msg = nullptr;                                                   \
    int rc = sqlite3_exec(database, sql, callback, nullptr, &err_msg);         \
    if (rc != SQLITE_OK) {                                                     \
      printf("SQLite error [%s, line %i]: %s\n", __FILE__, __LINE__, err_msg); \
      sqlite3_free(err_msg);                                                   \
    }                                                                          \
  }

}  // end namespace colmap

#endif  // COLMAP_SRC_UTIL_SQLITE3_UTILS_

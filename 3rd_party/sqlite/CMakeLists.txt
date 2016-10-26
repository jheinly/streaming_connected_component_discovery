cmake_minimum_required(VERSION 3.0)

cmh_add_library_module(
  src/sqlite/sqlite3.h
  src/sqlite/sqlite3ext.h
  src/sqlite/sqlite3.c
)

# SQLITE_THREADSAFE=0 - no mutexes, not thread-safe
# SQLITE_THREADSAFE=1 - mutexes enabled, serialized
# SQLITE_THREADSAFE=2 - mutexes enabled, multi-threaded (as long as each thread uses its own connection)
cmh_target_compile_definitions(
  -DSQLITE_THREADSAFE=2
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)

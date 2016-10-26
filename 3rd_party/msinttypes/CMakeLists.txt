cmake_minimum_required(VERSION 3.0)

set(INCLUDE_DIRECTORIES_LIST
  ${CMAKE_CURRENT_LIST_DIR}/src/inttypes
)

set(SOURCE_FILES
  src/inttypes/inttypes.h
)

if(MSVC60 OR MSVC70 OR MSVC71 OR MSVC80 OR MSVC90)
  list(APPEND INCLUDE_DIRECTORIES_LIST ${CMAKE_CURRENT_LIST_DIR}/src/stdint)
  set(SOURCE_FILES ${SOURCE_FILES} src/stdint/stdint.h)
  message(STATUS "Microsoft compiler older than 2010 detected, using msinttypes for stdint.h")
else()
  message(STATUS "Microsoft compiler equal to or newer than 2010 detected, using the system-provided stdint.h")
endif()

cmh_add_header_module(${SOURCE_FILES})

cmh_target_include_directories(
  ${INCLUDE_DIRECTORIES_LIST}
)

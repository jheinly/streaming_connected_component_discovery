cmake_minimum_required(VERSION 3.0)

cmh_add_header_module(
  src/assert/assert.h
)

set(ASSERT_ENABLED ON CACHE BOOL "Whether or not ASSERT() statements should be enabled.")

if(ASSERT_ENABLED)
  cmh_target_compile_definitions(
    -DASSERT_ENABLED
  )
endif()

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)

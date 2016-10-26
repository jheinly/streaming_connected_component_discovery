cmake_minimum_required(VERSION 3.0)

cmh_add_library_module(
  src/common/scoped_ptr.h
  src/common/windows/string_utils-inl.h
  src/common/windows/guid_string.h
  src/common/windows/guid_string.cc
  src/client/windows/common/ipc_protocol.h
  src/client/windows/handler/exception_handler.h
  src/client/windows/handler/exception_handler.cc
  src/client/windows/crash_generation/crash_generation_client.h
  src/client/windows/crash_generation/crash_generation_client.cc
  src/google_breakpad/common/breakpad_types.h
  src/google_breakpad/common/minidump_format.h
)

cmh_target_compile_definitions(
  UNICODE
  _UNICODE
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)

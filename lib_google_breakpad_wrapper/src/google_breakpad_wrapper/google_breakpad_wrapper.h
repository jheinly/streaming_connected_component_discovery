#pragma once
#ifndef GOOGLE_BREAKPAD_WRAPPER_H
#define GOOGLE_BREAKPAD_WRAPPER_H

#include <client/windows/handler/exception_handler.h>
#include <string>

class GoogleBreakpadWrapper
{
  public:
    GoogleBreakpadWrapper(
      const std::string & crash_dump_folder);

    ~GoogleBreakpadWrapper();

  private:
    static bool breakpad_filter_callback(
      void * context,
      EXCEPTION_POINTERS * exinfo,
      MDRawAssertionInfo * assertion);

    static bool breakpad_minidump_callback(
      const wchar_t * dump_path,
      const wchar_t * minidump_id,
      void * context,
      EXCEPTION_POINTERS * exinfo,
      MDRawAssertionInfo * assertion,
      bool succeeded);

    google_breakpad::ExceptionHandler * m_exception_handler;
};

#endif // GOOGLE_BREAKPAD_WRAPPER_H

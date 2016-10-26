#include <google_breakpad_wrapper/google_breakpad_wrapper.h>
#include <iostream>

GoogleBreakpadWrapper::GoogleBreakpadWrapper(
  const std::string & crash_dump_folder)
: m_exception_handler(NULL)
{
  const std::wstring crash_dump_folder_w(
    crash_dump_folder.begin(),
    crash_dump_folder.end());

  m_exception_handler = new google_breakpad::ExceptionHandler(
    crash_dump_folder_w,
    GoogleBreakpadWrapper::breakpad_filter_callback,
    GoogleBreakpadWrapper::breakpad_minidump_callback,
    NULL, // callback_context,
    google_breakpad::ExceptionHandler::HANDLER_ALL,
    (MINIDUMP_TYPE)(MiniDumpNormal |
      MiniDumpWithDataSegs |
      MiniDumpWithPrivateReadWriteMemory |
      MiniDumpWithHandleData |
      MiniDumpWithFullMemoryInfo |
      MiniDumpWithProcessThreadData |
      MiniDumpWithThreadInfo |
      MiniDumpWithUnloadedModules |
      MiniDumpWithIndirectlyReferencedMemory |
      MiniDumpScanMemory),
    (HANDLE)NULL, // pipe_handle
    (const google_breakpad::CustomClientInfo *)NULL);
}

GoogleBreakpadWrapper::~GoogleBreakpadWrapper()
{
  if (m_exception_handler != NULL)
  {
    delete m_exception_handler;
    m_exception_handler = NULL;
  }
}

bool GoogleBreakpadWrapper::breakpad_filter_callback(
  void * /*context*/,
  EXCEPTION_POINTERS * /*exinfo*/,
  MDRawAssertionInfo * /*assertion*/)
{
  std::cerr << std::endl;
  std::cerr << "GoogleBreakpadWrapper: Attempting to write dump file..." << std::endl;
  std::cerr << std::endl;
  return true;
}

bool GoogleBreakpadWrapper::breakpad_minidump_callback(
  const wchar_t * /*dump_path*/,
  const wchar_t * /*minidump_id*/,
  void * /*context*/,
  EXCEPTION_POINTERS * /*exinfo*/,
  MDRawAssertionInfo * /*assertion*/,
  bool /*succeeded*/)
{
  std::cerr << std::endl;
  std::cerr << "GoogleBreakpadWrapper: Done writing dump file. Passing exception to OS..." << std::endl;
  std::cerr << std::endl;
  return false;
}

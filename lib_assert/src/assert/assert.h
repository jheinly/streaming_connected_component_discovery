#pragma once
#ifndef ASSERT_H
#define ASSERT_H

#include <iostream>
#include <string>
#include <cstdlib>
#include <exception>

#ifdef ASSERT_ENABLED
  #define ASSERT(expression) (void)((expression) || (assert_helper::assert_failed(#expression, __FILE__, __LINE__), 0))
#else
  #define ASSERT(expression)
#endif

#define ALWAYS_ASSERT(expression) (void)((expression) || (assert_helper::assert_failed(#expression, __FILE__, __LINE__), 0))

#define COMPILER_ASSERT(expression) { typedef char compiler_assert_failed[(expression) ? 1 : -1]; }

namespace assert_helper {

class AssertException : public std::exception
{
  public:
    AssertException(const std::string & message)
    : std::exception(),
      m_message(message)
    {}

    virtual const char * what() const throw()
    {
      return m_message.c_str();
    }

  private:
    std::string m_message;
};

inline void assert_failed(
  const std::string & expression,
  const std::string & filename,
  const int line_number)
{
  std::cerr << std::endl;
  std::cerr << "ERROR - Assertion failed: \"" << expression << "\"" << std::endl;
  std::cerr << "  file: " << filename << std::endl;
  std::cerr << "  line: " << line_number << std::endl;
  std::cerr << std::endl;
  std::cerr.flush();
  throw(AssertException(expression));
}

} // namespace assert_helper

#endif // ASSERT_H

#pragma once
#ifndef COMPILER_MACROS_H
#define COMPILER_MACROS_H

namespace core {

#define VISUAL_STUDIO_2008 1500
#define VISUAL_STUDIO_2010 1600
#define VISUAL_STUDIO_2012 1700
#define VISUAL_STUDIO_2013 1800

#ifdef _MSC_VER
  #define VISUAL_STUDIO_VERSION _MSC_VER
#endif

} // namespace core

#endif // COMPILER_MACROS_H

cmake_minimum_required(VERSION 3.0)

# Detect if this is a 64-bit build.
CMH_TEST_IF_64_BIT(64BIT)

# Set path to NASM.
# NASM can be downloaded from http://www.nasm.us/
if(WIN32)
  set(NASM ${CMAKE_CURRENT_LIST_DIR}/nasm/win32/nasm.exe)
elseif(APPLE)
  set(NASM ${CMAKE_CURRENT_LIST_DIR}/nasm/macosx/nasm)
  execute_process(COMMAND chmod 755 ${NASM})
elseif(UNIX)
  set(NASM nasm)
else()
  message(FATAL_ERROR "Download NASM for current operating system and set path in libjpeg_turbo CMakeLists.txt file.")
endif()

# Set the flags for NASM.
# Example flag configurations can be found in the original libjpeg-turbo/configure file, and search for references to NAFLAGS.
if(64BIT)
  if(WIN32)
    set(NASM_FLAGS
      -fwin64 -DWIN64 -D__x86_64__
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/win/
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/simd/
    )
  elseif(UNIX)
    set(NASM_FLAGS
      -fmacho64 -DMACHO -D__x86_64__
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/win/
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/simd/
    )
  else()
    message(FATAL_ERROR "Set NASM flags for 64-bit on current operating system in libjpeg_turbo CMakeLists.txt file.")
  endif()
else()
  if(WIN32)
    set(NASM_FLAGS
      -fwin32 -DWIN32
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/win/
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/simd/
    )
  elseif(UNIX)
    set(NASM_FLAGS
      -fmacho -DMACHO
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/win/
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/simd/
    )
  else()
    message(FATAL_ERROR "Set NASM flags for 32-bit on current operating system in libjpeg_turbo CMakeLists.txt file.")
  endif()
endif()

# Settings for the config files.
set(VERSION 1.3.0)
set(JPEG_LIB_VERSION 62)
set(BUILD 1)
set(C_ARITH_CODING_SUPPORTED 1)
set(D_ARITH_CODING_SUPPORTED 1)
set(MEM_SRCDST_SUPPORTED 1)

# Generate header files from the config files.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/win/jconfig.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/jconfig.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/win/config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/config.h)

# Main libjpeg source files.
set(JPEG_SOURCE_FILES
  src/libjpeg_turbo/jchuff.h
  src/libjpeg_turbo/jdct.h
  src/libjpeg_turbo/jdhuff.h
  src/libjpeg_turbo/jerror.h
  src/libjpeg_turbo/jinclude.h
  src/libjpeg_turbo/jmemsys.h
  src/libjpeg_turbo/jmorecfg.h
  src/libjpeg_turbo/jpegcomp.h
  src/libjpeg_turbo/jpegint.h
  src/libjpeg_turbo/jpeglib.h
  src/libjpeg_turbo/jsimd.h
  src/libjpeg_turbo/jsimddct.h
  src/libjpeg_turbo/jversion.h
  src/libjpeg_turbo/jcapimin.c
  src/libjpeg_turbo/jcapistd.c
  src/libjpeg_turbo/jccoefct.c
  src/libjpeg_turbo/jccolor.c
  src/libjpeg_turbo/jcdctmgr.c
  src/libjpeg_turbo/jchuff.c
  src/libjpeg_turbo/jcinit.c
  src/libjpeg_turbo/jcmainct.c
  src/libjpeg_turbo/jcmarker.c
  src/libjpeg_turbo/jcmaster.c
  src/libjpeg_turbo/jcomapi.c
  src/libjpeg_turbo/jcparam.c
  src/libjpeg_turbo/jcphuff.c
  src/libjpeg_turbo/jcprepct.c
  src/libjpeg_turbo/jcsample.c
  src/libjpeg_turbo/jctrans.c
  src/libjpeg_turbo/jdapimin.c
  src/libjpeg_turbo/jdapistd.c
  src/libjpeg_turbo/jdatadst.c
  src/libjpeg_turbo/jdatasrc.c
  src/libjpeg_turbo/jdcoefct.c
  src/libjpeg_turbo/jdcolor.c
  src/libjpeg_turbo/jddctmgr.c
  src/libjpeg_turbo/jdhuff.c
  src/libjpeg_turbo/jdinput.c
  src/libjpeg_turbo/jdmainct.c
  src/libjpeg_turbo/jdmarker.c
  src/libjpeg_turbo/jdmaster.c
  src/libjpeg_turbo/jdmerge.c
  src/libjpeg_turbo/jdphuff.c
  src/libjpeg_turbo/jdpostct.c
  src/libjpeg_turbo/jdsample.c
  src/libjpeg_turbo/jdtrans.c
  src/libjpeg_turbo/jerror.c
  src/libjpeg_turbo/jfdctflt.c
  src/libjpeg_turbo/jfdctfst.c
  src/libjpeg_turbo/jfdctint.c
  src/libjpeg_turbo/jidctflt.c
  src/libjpeg_turbo/jidctfst.c
  src/libjpeg_turbo/jidctint.c
  src/libjpeg_turbo/jidctred.c
  src/libjpeg_turbo/jmemmgr.c
  src/libjpeg_turbo/jmemnobs.c
  src/libjpeg_turbo/jquant1.c
  src/libjpeg_turbo/jquant2.c
  src/libjpeg_turbo/jutils.c
)

# Arithmetic encoding/decoding support.
set(JPEG_SOURCE_FILES
  ${JPEG_SOURCE_FILES}
  src/libjpeg_turbo/jaricom.c
  src/libjpeg_turbo/jcarith.c
  src/libjpeg_turbo/jdarith.c
)

# Set the names of the assembly SIMD files.
if(64BIT)
  set(SIMD_BASENAMES
    jccolss2-64
    jcgrass2-64
    jcqnts2f-64
    jcqnts2i-64
    jcsamss2-64
    jdcolss2-64
    jdmerss2-64
    jdsamss2-64
    jfss2fst-64
    jfss2int-64
    jfsseflt-64
    jiss2flt-64
    jiss2fst-64
    jiss2int-64
    jiss2red-64
  )
else()
  set(SIMD_BASENAMES
    jccolmmx
    jccolss2
    jcgrammx
    jcgrass2
    jcqnt3dn
    jcqntmmx
    jcqnts2f
    jcqnts2i
    jcqntsse
    jcsammmx
    jcsamss2
    jdcolmmx
    jdcolss2
    jdmermmx
    jdmerss2
    jdsammmx
    jdsamss2
    jf3dnflt
    jfmmxfst
    jfmmxint
    jfss2fst
    jfss2int
    jfsseflt
    ji3dnflt
    jimmxfst
    jimmxint
    jimmxred
    jiss2flt
    jiss2fst
    jiss2int
    jiss2red
    jisseflt
    jsimdcpu
  )
endif()

# SIMD include files.
set(SIMD_INC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/simd/jcolsamp.inc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/simd/jdct.inc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/simd/jsimdext.inc
)

# Set the output directory for the SIMD obj files.
if(MSVC_IDE)
  set(OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}")
else()
  set(OBJ_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Set a SIMD source file based on 64-bit vs 32-bit.
if(64BIT)
  set(JPEG_SOURCE_FILES
    ${JPEG_SOURCE_FILES}
    src/libjpeg_turbo/simd/jsimd_x86_64.c
  )
else()
  set(JPEG_SOURCE_FILES
    ${JPEG_SOURCE_FILES}
    src/libjpeg_turbo/simd/jsimd_i386.c
  )
endif()

# Create the custom commands to compile the SIMD files.
foreach(SIMD_FILE ${SIMD_BASENAMES})
  set(DEP_FILE "")
  set(SIMD_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/simd/${SIMD_FILE}.asm)
  if(${SIMD_FILE} MATCHES col)
    set(DEP_FILE ${SIMD_FILE})
    string(REGEX REPLACE "col" "clr" DEP_FILE ${DEP_FILE})
    set(DEP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/simd/${DEP_FILE}.asm)
  endif()
  if(${SIMD_FILE} MATCHES mer)
    set(DEP_FILE ${SIMD_FILE})
    string(REGEX REPLACE "mer" "mrg" DEP_FILE ${DEP_FILE})
    set(DEP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/simd/${DEP_FILE}.asm)
  endif()
  if(${SIMD_FILE} MATCHES gra)
    set(DEP_FILE ${SIMD_FILE})
    string(REGEX REPLACE "gra" "gry" DEP_FILE ${DEP_FILE})
    set(DEP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/libjpeg_turbo/simd/${DEP_FILE}.asm)
  endif()
  set(SIMD_OBJ ${OBJ_DIR}/${SIMD_FILE}.obj)
  add_custom_command(OUTPUT ${SIMD_OBJ}
    DEPENDS ${SIMD_SRC} ${DEP_FILE} ${INC_FILES}
    COMMAND ${NASM} ${NASM_FLAGS} ${SIMD_SRC} -o${SIMD_OBJ})
  set(SIMD_OBJS ${SIMD_OBJS} ${SIMD_OBJ})
endforeach()

# Create a custom target to compile the SIMD files.
add_custom_target(libjpeg_turbo_simd DEPENDS ${SIMD_OBJS})

# This tells CMake that the source files haven't been generated yet.
set_source_files_properties(${SIMD_OBJS} PROPERTIES GENERATED 1)

source_group(${CMH_MODULE_NAME} FILES ${JPEG_SOURCE_FILES})

add_library(${CMH_MODULE_NAME} ${JPEG_SOURCE_FILES} ${SIMD_OBJS})

cmh_target_compile_definitions(
  -DWITH_SIMD
)

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${CMAKE_CURRENT_LIST_DIR}/src/libjpeg_turbo
)

add_dependencies(${CMH_MODULE_NAME} libjpeg_turbo_simd)

cmh_end_module()

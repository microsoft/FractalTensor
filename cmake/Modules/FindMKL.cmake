# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

if(MKL_INCLUDE_DIRS
   AND MKL_LIBRARIES
   AND MKL_INTERFACE_LIBRARY
   AND MKL_SEQUENTIAL_LAYER_LIBRARY
   AND MKL_CORE_LIBRARY)
  set(MKL_FIND_QUIETLY TRUE)
endif()

if(NOT BUILD_SHARED_LIBS)
  set(INT_LIB "libmkl_intel_ilp64.a")
  set(SEQ_LIB "libmkl_sequential.a")
  set(THR_LIB "libmkl_intel_thread.a")
  set(COR_LIB "libmkl_core.a")
else()
  set(INT_LIB "mkl_intel_ilp64")
  set(SEQ_LIB "mkl_sequential")
  set(THR_LIB "mkl_intel_thread")
  set(COR_LIB "mkl_core")
endif()

if(MSVC)
  set(ProgramFilesx86 "ProgramFiles(x86)")
  set(INTEL_ROOT_DEFAULT
      $ENV{${ProgramFilesx86}}/IntelSWTools/compilers_and_libraries/windows)
else()
  set(INTEL_ROOT_DEFAULT "/opt/intel")
endif()
set(INTEL_ROOT
    ${INTEL_ROOT_DEFAULT}
    CACHE PATH "Folder contains intel libs")
find_path(
  MKL_ROOT include/mkl.h
  PATHS $ENV{MKLROOT} ${INTEL_ROOT}/mkl
  DOC "Folder contains MKL")

find_path(
  MKL_INCLUDE_DIR
  NAMES mkl.h
  HINTS ${MKL_ROOT}/include /usr/include/mkl)

find_library(
  MKL_INTERFACE_LIBRARY
  NAMES ${INT_LIB}
  PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64 ${MKL_ROOT}/lib/intel64_win
        ${INTEL_ROOT}/mkl/lib/intel64)

find_library(
  MKL_SEQUENTIAL_LAYER_LIBRARY
  NAMES ${SEQ_LIB}
  PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64 ${INTEL_ROOT}/mkl/lib/intel64)

find_library(
  MKL_CORE_LIBRARY
  NAMES ${COR_LIB}
  PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64 ${INTEL_ROOT}/mkl/lib/intel64)

set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
set(MKL_LIBRARIES ${MKL_INTERFACE_LIBRARY} ${MKL_SEQUENTIAL_LAYER_LIBRARY}
                  ${MKL_CORE_LIBRARY})

# Added -Wl block to avoid circular dependencies.
# https://stackoverflow.com/questions/5651869/what-are-the-start-group-and- end-
# group-command-line-options https://software.intel.com/en- us/articles/intel-
# mkl-link-line-advisor
set(MKL_LIBRARIES -Wl,--start-group ${MKL_LIBRARIES} -Wl,--end-group)

# message("1 ${MKL_INCLUDE_DIR}") message("2 ${MKL_INTERFACE_LIBRARY}")
# message("3 ${MKL_SEQUENTIAL_LAYER_LIBRARY}") message("4 ${MKL_CORE_LIBRARY}")

if(MKL_INCLUDE_DIR
   AND MKL_INTERFACE_LIBRARY
   AND MKL_SEQUENTIAL_LAYER_LIBRARY
   AND MKL_CORE_LIBRARY)

  if(NOT DEFINED ENV{CRAY_PRGENVPGI}
     AND NOT DEFINED ENV{CRAY_PRGENVGNU}
     AND NOT DEFINED ENV{CRAY_PRGENVCRAY}
     AND NOT DEFINED ENV{CRAY_PRGENVINTEL}
     AND NOT MSVC)
    set(ABI "-m64")
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMKL_ILP64 ${ABI}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 ${ABI}")

else()
  set(MKL_INCLUDE_DIRS "")
  set(MKL_LIBRARIES "")
  set(MKL_INTERFACE_LIBRARY "")
  set(MKL_SEQUENTIAL_LAYER_LIBRARY "")
  set(MKL_CORE_LIBRARY "")

endif()

# Handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE if all
# listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  MKL
  DEFAULT_MSG
  MKL_LIBRARIES
  MKL_INCLUDE_DIRS
  MKL_INTERFACE_LIBRARY
  MKL_SEQUENTIAL_LAYER_LIBRARY
  MKL_CORE_LIBRARY)

mark_as_advanced(MKL_INCLUDE_DIRS MKL_LIBRARIES MKL_INTERFACE_LIBRARY
                 MKL_SEQUENTIAL_LAYER_LIBRARY MKL_CORE_LIBRARY)

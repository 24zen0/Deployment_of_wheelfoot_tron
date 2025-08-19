# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_mrosbridger_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED mrosbridger_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(mrosbridger_FOUND FALSE)
  elseif(NOT mrosbridger_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(mrosbridger_FOUND FALSE)
  endif()
  return()
endif()
set(_mrosbridger_CONFIG_INCLUDED TRUE)

# output package information
if(NOT mrosbridger_FIND_QUIETLY)
  message(STATUS "Found mrosbridger: 0.9.7 (${mrosbridger_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'mrosbridger' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${mrosbridger_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(mrosbridger_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "ament_cmake_export_include_directories-extras.cmake;ament_cmake_export_libraries-extras.cmake")
foreach(_extra ${_extras})
  include("${mrosbridger_DIR}/${_extra}")
endforeach()

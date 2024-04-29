cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(PROJECT_NAME FTorch)
set(LIB_NAME ftorch)
set(PACKAGE_VERSION 0.1)

project(${PROJECT_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES C CXX Fortran)

include(FortranCInterface)
FortranCInterface_VERIFY(CXX QUIET)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
if(ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(WARNING "No CUDA support")
  endif()
endif()

# Set RPATH behaviour
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
# Embed absolute paths to external libraries that are not part of
# the project, (they are expected to be at the same location on all
# machines the project will be deployed to
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Follow GNU conventions for installing directories
include(GNUInstallDirs)

# Define RPATH for executables via a relative expression to enable a
# fully relocatable package
file(RELATIVE_PATH relDir
 ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
 ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH $ORIGIN/${relDir})

# Dependencies
find_package(Torch REQUIRED)

# Library with C and Fortran bindings
add_subdirectory(src)

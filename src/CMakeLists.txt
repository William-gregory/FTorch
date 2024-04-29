add_library(${LIB_NAME} SHARED ctorch.cpp ftorch.f90)
# Add an alias FTorch::ftorch for the library
add_library(${PROJECT_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})
set_target_properties(${LIB_NAME} PROPERTIES
  PUBLIC_HEADER "ctorch.h"
  Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/modules"
  )
# Link TorchScript
target_link_libraries(${LIB_NAME} PRIVATE ${TORCH_LIBRARIES})
# Include the Fortran mod files in the library
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/modules>
#    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

# Install library, create target file
install(TARGETS "${LIB_NAME}"
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}
  )

# Install target file
install(EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Config.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

# Install Fortran module files
install(FILES "${CMAKE_BINARY_DIR}/modules/ftorch.mod"
 DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}"
 )

cmake_minimum_required(VERSION 2.8.3)
project(exotica_nlopt_solver)

find_package(catkin REQUIRED COMPONENTS
  exotica_core
)

include(ExternalProject)
ExternalProject_Add(
  NLopt
  GIT_REPOSITORY https://github.com/stevengj/NLopt.git
  GIT_TAG "master"
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  TEST_COMMAND ""
  CMAKE_ARGS -DNLOPT_PYTHON:BOOL=OFF -DNLOPT_OCTAVE:BOOL=OFF -DNLOPT_MATLAB:BOOL=OFF -DNLOPT_GUILE:BOOL=OFF -DNLOPT_SWIG:BOOL=OFF -DNLOPT_TESTS:BOOL=OFF -DCMAKE_INSTALL_PREFIX:PATH=${PROJECT_BINARY_DIR}/install -DCMAKE_BUILD_TYPE=Release
)
# TODO: copy (!)

AddInitializer(
  NLoptMotionSolver
  NLoptBoundedEndPoseSolver
  NLoptEndPoseSolver
  NLoptUnconstrainedEndPoseSolver
)
GenInitializers()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES exotica_nlopt_solver nlopt
  CATKIN_DEPENDS exotica_core
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

include_directories(${PROJECT_BINARY_DIR}/install/include)
link_directories(${PROJECT_BINARY_DIR}/install/lib)

add_library(${PROJECT_NAME}
  src/exotica_nlopt_solver.cpp
)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_initializers ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} NLopt)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} nlopt)

# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

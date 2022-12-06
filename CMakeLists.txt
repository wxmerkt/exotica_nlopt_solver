cmake_minimum_required(VERSION 3.0.2)
project(exotica_nlopt_solver)

find_package(catkin REQUIRED COMPONENTS
  exotica_core
)

# We can include the bleeding edge NLopt using ExternalProject. For Noetic, there is little advantage of doing so.
# include(ExternalProject)
# ExternalProject_Add(
#   NLopt
#   GIT_REPOSITORY https://github.com/stevengj/NLopt.git
#   GIT_TAG "master"
#   UPDATE_COMMAND ""
#   PATCH_COMMAND ""
#   TEST_COMMAND ""
#   # NLOPT_CXX BUILD_SHARED_LIBS
#   CMAKE_ARGS -DNLOPT_PYTHON:BOOL=OFF -DNLOPT_OCTAVE:BOOL=OFF -DNLOPT_MATLAB:BOOL=OFF -DNLOPT_GUILE:BOOL=OFF -DNLOPT_SWIG:BOOL=OFF -DNLOPT_TESTS:BOOL=OFF -DCMAKE_INSTALL_PREFIX:PATH=${PROJECT_BINARY_DIR}/install -DCMAKE_BUILD_TYPE=Release
# )
# # TODO: copy (!)
# include_directories(${PROJECT_BINARY_DIR}/install/include)
# link_directories(${PROJECT_BINARY_DIR}/install/lib)

AddInitializer(
  NLoptMotionSolver
  NLoptBoundedEndPoseSolver
  NLoptEndPoseSolver
  NLoptUnconstrainedEndPoseSolver
)
GenInitializers()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES exotica_nlopt_solver #nlopt
  CATKIN_DEPENDS exotica_core
)

# Use system-installed nlopt
find_package(NLopt REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

###########
## Build ##
###########

add_library(${PROJECT_NAME}
src/exotica_nlopt_solver.cpp
)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_initializers ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS}) # NLopt)
target_link_libraries(${PROJECT_NAME} PUBLIC ${catkin_LIBRARIES}) #nlopt)
target_link_libraries(${PROJECT_NAME} PUBLIC NLopt::nlopt)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

#############
## Install ##
#############

# Library (solver)
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Solver plug-in definiton
install(FILES exotica_plugins.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

# Headers
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

# Examples: Launch file and configs
install(DIRECTORY
    resources
    launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Examples: Executables
catkin_install_python(
  PROGRAMS
    scripts/example_nlopt_ik_bounded
    scripts/example_nlopt_ik_constrained
    scripts/example_nlopt_ik_unconstrained
    scripts/example_nlopt_ik_velocity_constrained
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

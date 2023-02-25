cmake_minimum_required(VERSION 3.14)
project(
  TES_T
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(SOURCE_DIR src)

# compiler flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using intel
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qno-offload")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()


option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# required compiler features
find_package(HDF5 COMPONENTS C CXX HL)
find_package(MPI REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

if(HDF5_FOUND)
  set(HAVE_HDF5 TRUE)
endif()

#configure_file(${CMAKE_SOURCE_DIR}/config_cmake.h.in ${CMAKE_SOURCE_DIR}/config.h)

# do the following so that RPATH of executables is such that they can still find libraries, 
#regardless of whether the insall location is default. (as per https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling)
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")

set(bin_PROGRAMS leap_frog)

set(common_LINKLIB ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} Boost::program_options Threads::Threads ${EIGEN3_LIBRARIES})   #MPI::MPI_CXX)

set(common_INCLUDE $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/> ${CMAKE_BINARY_DIR}
  ${DEP_INC} ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/.. ${EIGEN3_INCLUDE_DIR}
  ${HDF5_INCLUDE_DIRS})

add_executable(leap_frog ${SOURCE_DIR}/leap_frog.cpp ${SOURCE_DIR}/utilities.cpp)

foreach(program ${bin_PROGRAMS})
    target_link_libraries(${program} ${common_LINKLIB})
    target_include_directories(${program} PUBLIC ${common_INCLUDE})
    install(TARGETS ${program} DESTINATION ${CMAKE_INSTALL_PREFIX})
endforeach()


#message(STATUS ${HDF5_CXX_INCLUDE_DIRS})


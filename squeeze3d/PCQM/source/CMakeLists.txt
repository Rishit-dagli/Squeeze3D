# Copyright (C) 2017, 2023, 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(PROJECT_OUTPUT_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/../test")
set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING "Release;Debug")

project(pc_error)

# Comment this block out, if no multithreading is wanted. @DT
find_package(OpenMP)
if (OPENMP_FOUND)
  message(STATUS "OpenMP found.")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# Specify -DBOOST_ROOT=/data/tian/software/boost_1_63_0 in cmake command line, if a particular Boost version is required
# Specify -DBUILD_WITH_BOOST_DYNAMIC_LINKING=ON in cmake command line, if dynamic linking is needed
if(BUILD_WITH_BOOST_DYNAMIC_LINKING AND WIN32)
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_STATIC OFF)
  set(Boost_USE_MULTITHREAD ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB")
else(BUILD_WITH_BOOST_DYNAMIC_LINKING AND WIN32)
  if(NOT SHARED_LIBS OR WIN32)
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_STATIC ON)
  endif(NOT SHARED_LIBS OR WIN32)
endif(BUILD_WITH_BOOST_DYNAMIC_LINKING AND WIN32)

find_package(Boost REQUIRED COMPONENTS program_options)

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message( FATAL_ERROR "Can not compile without Boost library, CMake will exit." )
endif()

execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
if (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7)
  # message(STATUS "Version >= 4.7!" ${GCC_VERSION})
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  # message(STATUS "Version < 4.7!" ${GCC_VERSION})
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++0x")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

file(GLOB_RECURSE PROJ_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)
file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/./")

if (MSVC)
  add_executable(pc_error ${SOURCE})
else()
  list(REMOVE_ITEM SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/stdafx.cpp")
  add_executable(pc_error ${SOURCE})
endif()

target_link_libraries(pc_error ${Boost_PROGRAM_OPTIONS_LIBRARY})
install(TARGETS pc_error DESTINATION ".")

set_target_properties(pc_error PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_OUTPUT_FOLDER})
set_target_properties(pc_error PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_OUTPUT_FOLDER})
set_target_properties(pc_error PROPERTIES DEBUG_POSTFIX "_d")

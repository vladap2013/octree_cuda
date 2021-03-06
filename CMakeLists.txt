cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(cudex LANGUAGES CXX CUDA)

enable_language(CUDA)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)

# ----- cudex library
if (EXISTS ${CMAKE_SOURCE_DIR}/local/cudex)
	add_subdirectory(local/cudex)
endif()

if (NOT TARGET cudex_lib)
	message("Downloading cudex")
	include(FetchContent)

	FetchContent_Declare(
	  cudex
	  GIT_REPOSITORY git@github.com:vladap2013/cudex.git
	  GIT_TAG master
	)

	FetchContent_MakeAvailable(cudex)
endif()


# ----- Main library

add_library(octree_cuda INTERFACE)
target_include_directories(octree_cuda INTERFACE include)
target_link_libraries(octree_cuda INTERFACE cudex_lib)

# ----- Setup tests

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	message("Top level project, setting project variables")

	set(CMAKE_CUDA_ARCHITECTURES 61 75)

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
endif()

if (NOT TARGET gtest_main)
	message("Downloading google test")
	include(FetchContent)

	FetchContent_Declare(
	  googletest
	  GIT_REPOSITORY https://github.com/google/googletest.git
	  GIT_TAG master
	)

	FetchContent_MakeAvailable(googletest)
endif()

# main test
add_executable(octree_cuda_tests
	tests/test_helpers.cpp.cu
	tests/test_octree.cpp.cu
)
target_link_libraries(octree_cuda_tests octree_cuda glog::glog gtest_main)

add_test(octree_cuda_tests octree_cuda_tests)

# point cloud test
add_executable(octree_cuda_test_point_clouds
	tests/test_octree_point_clouds.cpp.cu
)

target_link_libraries(octree_cuda_test_point_clouds octree_cuda glog::glog gtest_main)

target_compile_definitions(octree_cuda_test_point_clouds PUBLIC
	TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/point_clouds/"
)

add_test(octree_cuda_test_point_clouds octree_cuda_test_point_clouds)

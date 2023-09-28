cmake_minimum_required(VERSION 3.24)
project(polynomial_bdg)
enable_language(CXX CUDA)

###  Apply compiler flags
include(cmake/CompilerFlags.cmake)

### Print operating system properties ###
include(cmake/PrintHostInfo.cmake)


### Add all source files recursively ###
file(GLOB_RECURSE CUDA_SOURCES "source_code/*.cu")
file(GLOB_RECURSE CXX_SOURCES "source_code/*.cpp")
file(GLOB_RECURSE HEADERS "source_code/*.h")
add_executable(${PROJECT_NAME} ${CUDA_SOURCES} ${CXX_SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PRIVATE source_code)



### Find all dependencies
include(cmake/SetupDependencies.cmake)
target_link_libraries(${PROJECT_NAME} PRIVATE deps flags)

### Apply compile flags for parts of the code ###
if(ENABLE_HARTREE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HARTREE)
endif()

if(ENABLE_GAUGE_FIELD)
    target_compile_definitions(${PROJECT_NAME} PRIVATE GAUGE_FIELD)
endif()

if(ENABLE_ZEEMAN)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ZEEMAN)
endif()


include(cmake/PrintTargetInfo.cmake)
print_project_summary(${PROJECT_NAME})


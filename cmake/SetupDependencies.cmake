

find_package(CUDAToolkit 11     REQUIRED)
find_package(h5pp       1.11.0  REQUIRED)

if(NOT TARGET deps)
    add_library(deps INTERFACE)
endif()
target_link_libraries(deps INTERFACE h5pp::h5pp)

cmake_minimum_required(VERSION 3.16)
project(miniFE_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

execute_process(
    COMMAND /bin/bash -c "make generate_info"
    RESULT_VARIABLE RESULT_CODE
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_executable(main
    src/main.cpp
    src/YAML_Doc.cpp
    src/YAML_Element.cpp
    utils/BoxPartition.cpp
    utils/param_utils.cpp
    utils/utils.cpp
    utils/mytimer.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_compile_definitions(main PRIVATE
    MINIFE_SCALAR=double
    MINIFE_LOCAL_ORDINAL=int
    MINIFE_GLOBAL_ORDINAL=int
    MINIFE_RESTRICT=__restrict__
    MINIFE_CSR_MATRIX
    MINIFE_INFO=1
    MINIFE_KERNELS=0
)
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/fem
)
target_compile_options(main PRIVATE -O3)

add_custom_target(run
    COMMAND $<TARGET_FILE:main> nx=64 ny=64 nz=64
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

cmake_minimum_required(VERSION 3.16)
project(xsbench_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    Main.cpp
    Simulation.cpp
    io.cu
    GridInit.cu
    Materials.cu
    XSutils.cu
)
set_source_files_properties(io.cu GridInit.cu Materials.cu XSutils.cu PROPERTIES LANGUAGE CXX)
target_link_libraries(main PRIVATE Kokkos::kokkos m)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(main PRIVATE -O3)

add_custom_target(run
    COMMAND $<TARGET_FILE:main> -s large -m event -r 10
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

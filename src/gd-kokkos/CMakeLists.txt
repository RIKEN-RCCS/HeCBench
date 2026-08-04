cmake_minimum_required(VERSION 3.16)
project(gd_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    ../gd-cuda/utils.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../gd-cuda
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        gisette_scale
        0.0001
        10
        100
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

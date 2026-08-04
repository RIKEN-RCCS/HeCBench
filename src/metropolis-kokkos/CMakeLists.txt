cmake_minimum_required(VERSION 3.16)
project(metropolis_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        -l
        32
        11
        -t
        4.7
        0.1
        -a
        10
        2
        2000
        10
        -h
        1.0
        -z
        7919
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

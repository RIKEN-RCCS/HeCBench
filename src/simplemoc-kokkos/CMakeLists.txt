cmake_minimum_required(VERSION 3.16)
project(simplemoc_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../simplemoc-omp
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        -s
        5000000
        -e
        128
        -n
        10
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

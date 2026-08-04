cmake_minimum_required(VERSION 3.16)
project(su3_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../su3-omp
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main> -i 1000 -l 32 -t 128 -v 3 -w 1
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

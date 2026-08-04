cmake_minimum_required(VERSION 3.16)
project(d2q9_bgk_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../d2q9-bgk-omp/Inputs/input_256x256.params"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../d2q9-bgk-omp/test.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_LIST_DIR}/../d2q9-bgk-omp/test.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../d2q9-bgk-omp"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        ../d2q9-bgk-omp/Inputs/input_256x256.params
        ../d2q9-bgk-omp/Obstacles/obstacles_256x256.dat
        100
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

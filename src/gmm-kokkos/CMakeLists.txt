cmake_minimum_required(VERSION 3.16)
project(gmm_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../gmm-omp/data"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../gmm-omp/data.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_LIST_DIR}/../gmm-omp/data.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../gmm-omp"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        100
        ../gmm-omp/data
        output
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

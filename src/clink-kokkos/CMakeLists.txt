cmake_minimum_required(VERSION 3.16)
project(clink_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

if((NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/input.hpp"
    OR NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/weight_1.hpp"
    OR NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/weight_2.hpp")
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../clink-omp/data.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_CURRENT_LIST_DIR}/../clink-omp/data.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        10
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

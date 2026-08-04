cmake_minimum_required(VERSION 3.16)
project(ss_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../ss-omp
)

set(SS_DATA_DIR "${CMAKE_CURRENT_LIST_DIR}/../data/ss")
set(SS_INPUT "${SS_DATA_DIR}/StringSearch_Input.txt")
if(NOT EXISTS "${SS_INPUT}" AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../ss-sycl/input.tar.gz")
    file(MAKE_DIRECTORY "${SS_DATA_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_LIST_DIR}/../ss-sycl/input.tar.gz"
        WORKING_DIRECTORY "${SS_DATA_DIR}"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        ../data/ss/StringSearch_Input.txt
        FACGBHEDIFJKFACGB
        100
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

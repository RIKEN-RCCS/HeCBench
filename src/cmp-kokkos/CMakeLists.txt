cmake_minimum_required(VERSION 3.16)
project(cmp_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../cmp-cuda/data.tar.bz" AND
   NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../cmp-cuda/data/simple-synthetic.su")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xjf "${CMAKE_CURRENT_LIST_DIR}/../cmp-cuda/data.tar.bz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../cmp-cuda"
        RESULT_VARIABLE CMP_DATA_EXTRACT_RESULT
    )
    if(NOT CMP_DATA_EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract ../cmp-cuda/data.tar.bz")
    endif()
endif()

add_executable(main
    main.cpp
    ../cmp-cuda/log.cpp
    ../cmp-cuda/parser.cpp
    ../cmp-cuda/reference.cpp
    ../cmp-cuda/su_cdp.cpp
    ../cmp-cuda/su_gather.cpp
    ../cmp-cuda/su_trace.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../cmp-cuda
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        -c0
        1.98e-7
        -c1
        1.77e-6
        -nc
        5
        -aph
        600
        -tau
        0.002
        -v
        2
        -i
        ../cmp-cuda/data/simple-synthetic.su
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

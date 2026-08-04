cmake_minimum_required(VERSION 3.16)
project(henry_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../henry-cuda/IRMOF-1.cssr"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../henry-cuda/data.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_CURRENT_LIST_DIR}/../henry-cuda/data.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../henry-cuda"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        ../henry-cuda/IRMOF-1.cssr
        1000
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

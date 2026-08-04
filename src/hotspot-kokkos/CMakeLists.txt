cmake_minimum_required(VERSION 3.16)
project(hotspot_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

if((NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../data/hotspot/temp_512"
    OR NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../data/hotspot/power_512")
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../data/hotspot/hotspot.tar.bz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_CURRENT_LIST_DIR}/../data/hotspot/hotspot.tar.bz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../data/hotspot"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main> 512 2 100 ../data/hotspot/temp_512 ../data/hotspot/power_512 output.out
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

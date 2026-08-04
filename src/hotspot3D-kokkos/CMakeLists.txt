cmake_minimum_required(VERSION 3.16)
project(hotspot3D_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

if((NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../data/hotspot3D/power_512x8"
    OR NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../data/hotspot3D/temp_512x8")
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../data/hotspot3D/hotspot3D.tar.bz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xjf "${CMAKE_CURRENT_LIST_DIR}/../data/hotspot3D/hotspot3D.tar.bz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../data/hotspot3D"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        512
        8
        5000
        ../data/hotspot3D/power_512x8
        ../data/hotspot3D/temp_512x8
        output.out
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

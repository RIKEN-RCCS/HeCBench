cmake_minimum_required(VERSION 3.16)
project(geodesic_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/locations.txt"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../geodesic-sycl/locations.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_LIST_DIR}/../geodesic-sycl/locations.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main> 1000
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

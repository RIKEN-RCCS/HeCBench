cmake_minimum_required(VERSION 3.16)
project(aes_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    reference.cpp
    utils.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../include
)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../urng-sycl/URNG_Input.bmp"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../urng-sycl/image.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_LIST_DIR}/../urng-sycl/image.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../urng-sycl"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        100
        0
        ../urng-sycl/URNG_Input.bmp
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

cmake_minimum_required(VERSION 3.16)
project(sobel_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_compile_options(main PRIVATE -O3)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../sobel-sycl/SobelFilter_Input.bmp"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../sobel-sycl/data.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_CURRENT_LIST_DIR}/../sobel-sycl/data.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../sobel-sycl"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main> ../sobel-sycl/SobelFilter_Input.bmp 1000
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

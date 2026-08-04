cmake_minimum_required(VERSION 3.16)
project(minimap2_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/in-1k.txt"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../minimap2-sycl/in-1k.txt.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_LIST_DIR}/../minimap2-sycl/in-1k.txt.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
        RESULT_VARIABLE minimap2_data_result
    )
    if(NOT minimap2_data_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract ../minimap2-sycl/in-1k.txt.tar.gz")
    endif()
endif()

add_executable(main
    device/device_kernel_wrapper.cpp
    src/host_data_io.cpp
    src/host_kernel.cpp
    src/main.cpp
    src/memory_scheduler.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ./include
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        in-1k.txt
        kernel-out.txt
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

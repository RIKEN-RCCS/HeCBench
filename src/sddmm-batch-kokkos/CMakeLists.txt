cmake_minimum_required(VERSION 3.16)
project(sddmm_batch_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        32
        1024
        1024
        1024
        1024
        1000
        1
    COMMAND $<TARGET_FILE:main>
        32
        4096
        4096
        4096
        4096
        1000
        0
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

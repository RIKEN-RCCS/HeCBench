cmake_minimum_required(VERSION 3.16)
project(laplace3d_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        128
        128
        128
        100
        1
    COMMAND $<TARGET_FILE:main> 512 512 512 100 0
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

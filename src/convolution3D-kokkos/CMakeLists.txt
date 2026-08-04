cmake_minimum_required(VERSION 3.16)
project(convolution3D_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        32 1 6 32 32 5 100
    COMMAND $<TARGET_FILE:main> 32 6 16 14 14 5 100
    COMMAND $<TARGET_FILE:main> 32 96 256 26 26 5 100
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

cmake_minimum_required(VERSION 3.16)
project(log2_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        ../log2-cuda/log2_parameters.config
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

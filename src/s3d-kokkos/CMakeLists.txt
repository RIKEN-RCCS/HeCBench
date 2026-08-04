cmake_minimum_required(VERSION 3.16)
project(s3d_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    s3d.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../s3d-omp
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        2
        100
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

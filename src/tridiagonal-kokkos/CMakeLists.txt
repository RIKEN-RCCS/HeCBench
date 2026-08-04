cmake_minimum_required(VERSION 3.16)
project(tridiagonal_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    shrUtils.cpp
    cmd_arg_reader.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_compile_definitions(main PRIVATE
    UNIX
    BENCH_ITERATIONS=100
)
add_custom_target(run
    COMMAND $<TARGET_FILE:main>
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

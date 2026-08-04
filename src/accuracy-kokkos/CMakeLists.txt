cmake_minimum_required(VERSION 3.16)
project(accuracy_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../accuracy-cuda
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        8192
        10000
        10
        100
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

cmake_minimum_required(VERSION 3.16)
project(lebesgue_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../lebesgue-cuda
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        1000000
        2
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

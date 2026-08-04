cmake_minimum_required(VERSION 3.16)
project(surfel_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        1024
        640
        480
        100
    COMMAND $<TARGET_FILE:main> 30000 640 480 100
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

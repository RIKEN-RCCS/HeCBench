cmake_minimum_required(VERSION 3.16)
project(srad_kokkos LANGUAGES C CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    srad.cpp
    util/graphics/graphics.c
    util/graphics/resize.c
    util/timer/timer.c
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_link_libraries(main PRIVATE m)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    .
    ./util/graphics
    ./util/timer
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        1000
        0.5
        502
        458
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

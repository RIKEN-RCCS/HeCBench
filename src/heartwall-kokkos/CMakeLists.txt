cmake_minimum_required(VERSION 3.16)
project(heartwall_kokkos LANGUAGES C CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    util/avi/avilib.c
    util/avi/avimod.c
    util/file/file.c
    util/timer/timer.c
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_link_libraries(main PRIVATE m)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ./util/avi
    ./util/file
    ./util/timer
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        ../data/heartwall/test.avi
        20
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

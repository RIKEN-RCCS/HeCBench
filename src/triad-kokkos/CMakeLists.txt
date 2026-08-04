cmake_minimum_required(VERSION 3.16)
project(triad_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    Option.cpp
    OptionParser.cpp
    Timer.cpp
    main.cpp
    triad.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

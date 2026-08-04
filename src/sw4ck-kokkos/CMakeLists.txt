cmake_minimum_required(VERSION 3.16)
project(sw4ck_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    curvilinear4sg.cpp
    utils.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../sw4ck-omp
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        ../sw4ck-cuda/sw4ck.in
        100
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

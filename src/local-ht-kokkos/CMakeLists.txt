cmake_minimum_required(VERSION 3.16)
project(local_ht_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        ../local-ht-cuda/locassm_data/localassm_extend_7-21.dat
        21
        ./out_file.dat
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

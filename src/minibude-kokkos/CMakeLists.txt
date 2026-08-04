cmake_minimum_required(VERSION 3.16)
project(minibude_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E tar xzf ../minibude-sycl/data.tar.gz
    COMMAND $<TARGET_FILE:main>
        --deck
        data/bm1
        --wgsize
        256
        --iterations
        8
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

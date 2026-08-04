cmake_minimum_required(VERSION 3.16)
project(face_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    haar.cpp
    image.cpp
    stdio-wrapper.cpp
    rectangles.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        ../face-cuda/Face.pgm
        ../face-cuda/info.txt
        ../face-cuda/class.txt
        Output.pgm
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

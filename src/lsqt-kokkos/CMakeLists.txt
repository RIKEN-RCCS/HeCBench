cmake_minimum_required(VERSION 3.16)
project(lsqt_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    lsqt.cpp
    sigma.cpp
    vector.cpp
    hamiltonian.cpp
    model.cpp
    model_general.cpp
    model_lattice.cpp
    anderson.cpp
    charge.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    .
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        ../lsqt-cuda/examples/input.txt
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

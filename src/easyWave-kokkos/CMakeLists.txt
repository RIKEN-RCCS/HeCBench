cmake_minimum_required(VERSION 3.16)
project(easyWave_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    easyWave.cpp
    cOgrd.cpp
    cOkadaEarthquake.cpp
    cOkadaFault.cpp
    cSphere.cpp
    utilits.cpp
    okada.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    .
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        -grid
        data/grids/e2Asean.grd
        -source
        data/faults/BengkuluSept2007.flt
        -time
        120
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

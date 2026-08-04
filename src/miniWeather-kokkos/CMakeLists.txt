cmake_minimum_required(VERSION 3.16)
project(miniWeather_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos MPI::MPI_CXX)
target_compile_features(main PRIVATE cxx_std_17)
target_compile_definitions(main PRIVATE
    NX=400
    NZ=200
    SIM_TIME=600
    DATA_SPEC=DATA_SPEC_THERMAL
)

add_custom_target(run
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:main>
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

cmake_minimum_required(VERSION 3.16)
project(xlqc_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)
find_package(Eigen3 REQUIRED)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/geom.dat"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../xlqc-cuda/example.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_LIST_DIR}/../xlqc-cuda/example.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
        RESULT_VARIABLE xlqc_data_result
    )
    if(NOT xlqc_data_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract ../xlqc-cuda/example.tar.gz")
    endif()
endif()

add_executable(main
    main.cpp
    ../xlqc-cuda/scf.cc
    ../xlqc-cuda/basis.cc
    ../xlqc-cuda/int_lib/crys.cc
    ../xlqc-cuda/int_lib/cints.cc
)
target_link_libraries(main PRIVATE Kokkos::kokkos Eigen3::Eigen)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../xlqc-cuda
    ../xlqc-cuda/int_lib
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        sp
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

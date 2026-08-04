cmake_minimum_required(VERSION 3.16)
project(deredundancy_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../deredundancy-sycl/testData.fasta"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../deredundancy-sycl/testData.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_CURRENT_LIST_DIR}/../deredundancy-sycl/testData.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../deredundancy-sycl"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        i
        ../deredundancy-sycl/testData.fasta
        o
        output.txt
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
